from __future__ import annotations

import argparse
import json

from deal_radar.engine import DealRadarEnsembleEngine
from deal_radar.models import DealOpportunity


def run_demo(min_discount_pct: float = 15.0, max_disagreement_pct: float = 25.0) -> list[dict]:
    engine = DealRadarEnsembleEngine(
        min_discount_pct=min_discount_pct,
        max_model_disagreement_pct=max_disagreement_pct,
    )
    samples = [
        {
            "deal": DealOpportunity(
                title="Sony WH-1000XM5 Wireless Headphones",
                url="deals.local/sony-xm5",
                merchant="ExampleStore",
                category="Audio",
                listed_price=249.0,
                shipping_cost=10.0,
                tax_estimate=15.0,
                in_stock=True,
                currency="USD",
            ),
            "frontier_price": 340.0,
            "specialist_price": 350.0,
            "neural_price": 330.0,
        },
        {
            "deal": DealOpportunity(
                title="Sony WH-1000XM5 Wireless Headphones (Duplicate)",
                url="deals.local/sony-xm5",
                merchant="ExampleStore",
                category="Audio",
                listed_price=249.0,
                shipping_cost=10.0,
                tax_estimate=15.0,
                in_stock=True,
                currency="USD",
            ),
            "frontier_price": 340.0,
            "specialist_price": 350.0,
            "neural_price": 330.0,
        },
        {
            "deal": DealOpportunity(
                title="4K OLED Monitor 27-inch",
                url="deals.local/oled-27",
                merchant="TechOutlet",
                category="Monitors",
                listed_price=520.0,
                shipping_cost=0.0,
                tax_estimate=40.0,
                in_stock=True,
                currency="USD",
            ),
            "frontier_price": 680.0,
            "specialist_price": None,
            "neural_price": 660.0,
        },
    ]

    results = []
    for item in samples:
        receipt = engine.evaluate_deal(
            deal=item["deal"],
            frontier_price=item["frontier_price"],
            specialist_price=item["specialist_price"],
            neural_price=item["neural_price"],
        )
        results.append(
            {
                "title": item["deal"].title,
                "url": receipt.deal_url,
                "should_notify": receipt.should_notify,
                "reason": receipt.reason,
                "landed_price": receipt.landed_price,
                "estimated_value": receipt.estimate.weighted_value,
                "discount_pct": receipt.discount_pct,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ensemble Deal Radar demo evaluation.")
    parser.add_argument("--min-discount-pct", type=float, default=15.0)
    parser.add_argument("--max-disagreement-pct", type=float, default=25.0)
    args = parser.parse_args()

    results = run_demo(
        min_discount_pct=args.min_discount_pct,
        max_disagreement_pct=args.max_disagreement_pct,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
