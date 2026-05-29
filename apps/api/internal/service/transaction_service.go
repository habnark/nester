package service

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/shopspring/decimal"

	"github.com/suncrestlabs/nester/apps/api/internal/domain/transaction"
)

type TransactionService struct {
	repository transaction.Repository
	horizonURL  string
	client      *http.Client
}

type RegisterTransactionInput struct {
	VaultID  uuid.UUID
	Type     transaction.TransactionType
	Amount   decimal.Decimal
	Currency string
	TxHash   string
}

func NewTransactionService(repository transaction.Repository, horizonURL string) *TransactionService {
	return &TransactionService{
		repository: repository,
		horizonURL: strings.TrimRight(strings.TrimSpace(horizonURL), "/"),
		client: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

func (s *TransactionService) RegisterTransaction(ctx context.Context, input RegisterTransactionInput) (transaction.Transaction, error) {
	if input.VaultID == uuid.Nil || input.Amount.Cmp(decimal.Zero) <= 0 || strings.TrimSpace(input.Currency) == "" || strings.TrimSpace(input.TxHash) == "" {
		return transaction.Transaction{}, transaction.ErrInvalidTransaction
	}
	normalizedType := transaction.TransactionType(strings.ToLower(strings.TrimSpace(string(input.Type))))
	if !isSupportedTransactionType(normalizedType) {
		return transaction.Transaction{}, transaction.ErrInvalidType
	}

	model := transaction.Transaction{
		ID:        uuid.New(),
		VaultID:   input.VaultID,
		Type:      normalizedType,
		Amount:    input.Amount,
		Currency:  strings.ToUpper(strings.TrimSpace(input.Currency)),
		TxHash:    strings.TrimSpace(input.TxHash),
		Status:    transaction.StatusPending,
		CreatedAt: time.Now().UTC(),
		UpdatedAt: time.Now().UTC(),
	}

	return s.repository.Upsert(ctx, model)
}

func (s *TransactionService) GetTransaction(ctx context.Context, hash string) (transaction.Transaction, error) {
	if strings.TrimSpace(hash) == "" {
		return transaction.Transaction{}, transaction.ErrInvalidTransaction
	}

	model, err := s.repository.GetByHash(ctx, hash)
	if err != nil {
		return transaction.Transaction{}, err
	}

	updated, _, err := s.ReconcileTransaction(ctx, model)
	if err != nil {
		return transaction.Transaction{}, err
	}
	return updated, nil
}

// ListPendingOlderThan returns transactions still pending whose age exceeds
// minAge. The background poller calls this each tick; minAge keeps freshly
// submitted transactions (which Horizon hasn't ingested yet) out of the batch.
func (s *TransactionService) ListPendingOlderThan(ctx context.Context, minAge time.Duration) ([]transaction.Transaction, error) {
	cutoff := time.Now().UTC().Add(-minAge)
	return s.repository.ListPendingOlderThan(ctx, cutoff)
}

// ReconcileTransaction checks the on-chain status of a single transaction
// against Horizon and persists a terminal status (completed/failed) if one has
// been reached. It returns the latest transaction view and whether the status
// actually changed. Transactions already in a terminal state, and those still
// pending on-chain, are returned unchanged with changed=false. This is the
// single source of truth for status reconciliation, shared by GetTransaction
// (on-demand) and the background poller.
func (s *TransactionService) ReconcileTransaction(ctx context.Context, model transaction.Transaction) (transaction.Transaction, bool, error) {
	switch model.Status {
	case transaction.StatusCompleted, transaction.StatusFailed:
		return model, false, nil
	}

	horizonStatus, confirmedAt, errorReason, err := s.lookupHorizonTransaction(ctx, model.TxHash)
	if err != nil {
		if errors.Is(err, errTransactionPending) {
			return model, false, nil
		}
		return model, false, err
	}

	switch horizonStatus {
	case transaction.StatusCompleted, transaction.StatusFailed:
		updated, updateErr := s.repository.UpdateStatus(ctx, model.TxHash, horizonStatus, confirmedAt, errorReason)
		if updateErr != nil {
			return model, false, updateErr
		}
		return updated, true, nil
	default:
		return model, false, nil
	}
}

type horizonTransactionResponse struct {
	Successful bool   `json:"successful"`
	CreatedAt  string `json:"created_at"`
	ResultXdr  string `json:"result_xdr"`
}

var errTransactionPending = errors.New("transaction pending")

func (s *TransactionService) lookupHorizonTransaction(ctx context.Context, hash string) (transaction.TransactionStatus, *time.Time, string, error) {
	if s.horizonURL == "" {
		return transaction.StatusPending, nil, "", nil
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("%s/transactions/%s", s.horizonURL, hash), nil)
	if err != nil {
		return "", nil, "", err
	}

	resp, err := s.client.Do(req)
	if err != nil {
		return "", nil, "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return transaction.StatusPending, nil, "", errTransactionPending
	}
	if resp.StatusCode != http.StatusOK {
		return "", nil, "", fmt.Errorf("horizon status lookup failed: %s", resp.Status)
	}

	var payload horizonTransactionResponse
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return "", nil, "", err
	}

	confirmedAt, err := time.Parse(time.RFC3339, payload.CreatedAt)
	if err != nil {
		return "", nil, "", err
	}

	if payload.Successful {
		return transaction.StatusCompleted, &confirmedAt, "", nil
	}

	return transaction.StatusFailed, &confirmedAt, strings.TrimSpace(payload.ResultXdr), nil
}

func isSupportedTransactionType(value transaction.TransactionType) bool {
	switch value {
	case transaction.TypeDeposit, transaction.TypeWithdrawal, transaction.TypeSettlement:
		return true
	default:
		return false
	}
}
