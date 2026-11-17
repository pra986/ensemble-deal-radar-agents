# Architecture: Ensemble Deal Radar Agents (Project 08)

## Overview
`DealRadarEnsembleEngine` combines three independent valuation signals into a weighted ensemble estimate, verifies true landed cost, filters out high-disagreement outliers, and deduplicates notifications by canonical listing URL.

## Ensemble Weights
- **Frontier Agent (`0.8`)**: High-reasoning LLM pricer using retrieval-augmented comparables.
- **QLoRA Specialist Agent (`0.1`)**: Fine-tuned 4-bit adapter pricer for domain-specific SKU valuation.
- **Neural / Tabular Pricer (`0.1`)**: Fast feature-based baseline estimator.

If the specialist agent times out or is unavailable, the engine re-normalizes the remaining weights (`0.8 / 0.9` frontier, `0.1 / 0.9` neural) and flags `specialist_fallback=True`.

## Production Guardrails
1. **Landed-Cost Verification**: Computes `landed_cost = listed_price + shipping_cost + estimated_tax` before comparing against the ensemble fair-value estimate.
2. **Model Disagreement Check**: Rejects alerts where `max_spread_pct > max_model_disagreement_pct` to suppress false positives on ambiguous listings.
3. **Seen-URL Deduplication**: Tracks normalized listing URLs in memory / state store so repeated scrapes never trigger duplicate Pushover notifications.
