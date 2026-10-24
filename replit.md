# StockPilot Seller Center

Inventory and storage management for small Shopee/Lazada sellers, with dated sales
history, simulated time, barcode movements and forecast-based resupply.

## Run & Operate

- Managed frontend: `artifacts/seller-center: web` (root preview).
- Managed backend: `artifacts/api-server: API Server` (`/api`).
- `pnpm run typecheck` — all TypeScript packages.
- `pnpm --filter @workspace/api-spec run codegen` — API hooks and validators.
- `pnpm --filter @workspace/db run push` — development schema changes.
- PostgreSQL is persistent. Production schema changes use the Publish flow.
- Python 3 is required by root `ml_service.py`; statistical mode uses stdlib only.

## Product behavior

- The requested demo account is `demo` / `password123`, store “Shopee & Lazada
  SuperStore”. It is seeded once with six products and 30 days of sales each.
  This account is shared demonstration data, not a private production store.
- Users can register separate username/password stores. Passwords use scrypt,
  sessions use random HTTP-only cookies and hashed tokens in PostgreSQL.
- The demo date controls storage age and forecast cutoff. It does not rewind
  current stock or undo real inventory movements.
- Historical edits default to no stock deduction. Opt-in deductions account for
  previously deducted units so identical repeated saves do not deduct twice.
- Inbound scans and combine receipts allocate arrivals to oldest pending orders
  first. Partial deliveries retain Delivering and the previous arrival date.
  Only clearing all pending stock sets In Storage and updates the arrival date.
- Resupply quantities cannot exceed capacity including pending orders.
- CSV import is atomic: all valid rows are saved or no rows are changed.

## Forecasts

See `models/README.md`. The uploaded weights are retained. Valid trained inference
also requires original product ordering and calendar metadata, which were not
provided in the upload. Statistical fallback is explicit in API/UI.
Do not silently remap new product barcodes to trained SKU embeddings.

## Structure

- `artifacts/seller-center/src/` — React interface.
- `artifacts/api-server/src/routes/` — authenticated, store-scoped APIs.
- `lib/db/src/schema/inventory.ts` — relational persistence.
- `lib/api-spec/openapi.yaml` — API contract.
- `ml_service.py`, `ml/trained_adapter.py` — forecast bridge and optional ensemble.
- `ml/original/` — unchanged uploaded training modules.
- `models/` — supplied weights and inference setup documentation.

## Code generation

The Orval version generates a path schema and query type with the same name for
history endpoints. The codegen script intentionally exports only the generated
Zod schema barrel for the server; DTOs are imported from the client package.
