# Ensemble Deal Radar Agents (Project 08)

Autonomous bargain-finding multi-agent ensemble (`0.8` frontier + `0.1` QLoRA specialist + `0.1` neural) with landed-cost verification and URL deduplication.

## Architecture
- **Frontier Agent (`0.8`)**: High-reasoning LLM pricer using retrieval-augmented comparables.
- **QLoRA Specialist Agent (`0.1`)**: Fine-tuned 4-bit adapter pricer for domain-specific SKU valuation.
- **Neural / Tabular Agent (`0.1`)**: Lightweight embedding/MLP estimator for fast baseline anchoring.
- **Landed-Cost & URL Deduplication Guard**: Normalizes canonical URLs (stripping tracking params) and verifies `landed_cost = item_price + shipping + tax` before computing `discount_pct`.

## 6-Stage Workflow
1. **Feed Ingestion**: Collect candidate deals from RSS/API deal feeds.
2. **URL Canonicalization & Deduplication**: Normalize URLs and drop already-alerted or duplicate listings.
3. **Structured Extraction**: Parse title, category, base price, shipping, and tax into `DealCandidate`.
4. **Ensemble Valuation**: Compute weighted fair-value estimate (`0.8 * frontier + 0.1 * qlora + 0.1 * neural`).
5. **Landed-Cost Verification**: Enforce `landed_cost < fair_value * (1 - min_discount)` and stock/region checks.
6. **Alert Dispatch**: Emit ranked `DealAlert` records with confidence and savings breakdown.

## Data Contract
- `DealCandidate`: `deal_id`, `title`, `url`, `category`, `listed_price_usd`, `shipping_usd`, `tax_usd`, `in_stock`
- `EnsembleEstimate`: `frontier_usd`, `qlora_usd`, `neural_usd`, `weighted_usd`
- `DealAlert`: `deal_id`, `canonical_url`, `landed_cost_usd`, `fair_value_usd`, `discount_pct`

## Production Controls
- Canonical URL deduplication with persistent seen-set.
- Landed-cost floor & outlier clipping against single-agent spikes.
- Minimum discount threshold and out-of-stock suppression.

## Local Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -v
```

## Attribution
Pinned attribution: adapted and hardened from [`ed-donner/llm_engineering`](https://github.com/ed-donner/llm_engineering) (`week8`).
 specialist, 0.1 neural) with landed-cost verification and URL deduplication.
