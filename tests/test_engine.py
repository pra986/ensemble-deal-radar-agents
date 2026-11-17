import pytest

from deal_radar.engine import DealRadarEnsembleEngine
from deal_radar.models import DealOpportunity


def _make_deal(
    url: str = "deals.local/item-1",
    listed_price: float = 200.0,
    shipping_cost: float = 20.0,
    tax_estimate: float = 10.0,
    in_stock: bool = True,
) -> DealOpportunity:
    return DealOpportunity(
        title="Test Gadget",
        url=url,
        merchant="TestMerchant",
        category="Electronics",
        listed_price=listed_price,
        shipping_cost=shipping_cost,
        tax_estimate=tax_estimate,
        in_stock=in_stock,
        currency="USD",
    )


def test_seen_url_deduplication():
    engine = DealRadarEnsembleEngine(min_discount_pct=15.0)
    deal = _make_deal(url="deals.local/unique-1", listed_price=200.0, shipping_cost=0.0, tax_estimate=0.0)

    first = engine.evaluate_deal(deal, frontier_price=300.0, specialist_price=300.0, neural_price=300.0)
    assert first.should_notify is True

    second = engine.evaluate_deal(deal, frontier_price=300.0, specialist_price=300.0, neural_price=300.0)
    assert second.should_notify is False
    assert "duplicate" in second.reason.lower()


def test_specialist_timeout_fallback_renormalizes_weights():
    engine = DealRadarEnsembleEngine()
    estimate = engine.estimate_value(frontier_price=900.0, specialist_price=None, neural_price=810.0)

    expected = round(900.0 * (0.8 / 0.9) + 810.0 * (0.1 / 0.9), 2)
    assert estimate.specialist_fallback_used is True
    assert estimate.weighted_value == pytest.approx(expected, abs=1e-2)


def test_landed_cost_threshold_math():
    engine = DealRadarEnsembleEngine(min_discount_pct=15.0)
    # Landed cost = 200 + 30 + 30 = 260 vs estimated value 300 -> 13.33% discount (< 15%)
    deal = _make_deal(url="deals.local/landed-fail", listed_price=200.0, shipping_cost=30.0, tax_estimate=30.0)
    receipt = engine.evaluate_deal(deal, frontier_price=300.0, specialist_price=300.0, neural_price=300.0)

    assert receipt.landed_price == pytest.approx(260.0)
    assert receipt.discount_pct == pytest.approx(13.33, abs=0.05)
    assert receipt.should_notify is False


def test_high_disagreement_false_positive_rejection():
    engine = DealRadarEnsembleEngine(min_discount_pct=10.0, max_model_disagreement_pct=25.0)
    deal = _make_deal(url="deals.local/high-disagreement", listed_price=150.0, shipping_cost=0.0, tax_estimate=0.0)

    # Disagreement: (450 - 150) / weighted_value > 25%
    receipt = engine.evaluate_deal(deal, frontier_price=300.0, specialist_price=450.0, neural_price=150.0)
    assert receipt.estimate.disagreement_pct > 25.0
    assert receipt.should_notify is False
    assert "disagreement" in receipt.reason.lower()
