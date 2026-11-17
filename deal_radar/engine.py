from __future__ import annotations

from deal_radar.models import DealOpportunity, EnsembleEstimate, NotificationReceipt


class DealRadarEnsembleEngine:
    FRONTIER_WEIGHT = 0.8
    SPECIALIST_WEIGHT = 0.1
    NEURAL_WEIGHT = 0.1

    def __init__(
        self,
        min_discount_pct: float = 15.0,
        max_model_disagreement_pct: float = 25.0,
    ) -> None:
        self.min_discount_pct = min_discount_pct
        self.max_model_disagreement_pct = max_model_disagreement_pct
        self.seen_urls: set[str] = set()

    @staticmethod
    def _normalize_url(url: str) -> str:
        return url.strip().rstrip("/").lower()

    def estimate_value(
        self,
        frontier_price: float,
        specialist_price: float | None,
        neural_price: float,
    ) -> EnsembleEstimate:
        if specialist_price is None:
            denom = self.FRONTIER_WEIGHT + self.NEURAL_WEIGHT
            weighted = (
                self.FRONTIER_WEIGHT * frontier_price
                + self.NEURAL_WEIGHT * neural_price
            ) / denom
            prices = [frontier_price, neural_price]
            fallback = True
        else:
            weighted = (
                self.FRONTIER_WEIGHT * frontier_price
                + self.SPECIALIST_WEIGHT * specialist_price
                + self.NEURAL_WEIGHT * neural_price
            )
            prices = [frontier_price, specialist_price, neural_price]
            fallback = False

        mean_price = sum(prices) / len(prices)
        spread_pct = ((max(prices) - min(prices)) / mean_price) * 100.0 if mean_price else 0.0
        return EnsembleEstimate(
            frontier_price=round(frontier_price, 2),
            specialist_price=round(specialist_price, 2) if specialist_price is not None else None,
            neural_price=round(neural_price, 2),
            weighted_estimate=round(weighted, 2),
            disagreement_pct=round(spread_pct, 2),
            specialist_fallback=fallback,
        )

    def evaluate_deal(
        self,
        opportunity: DealOpportunity,
        frontier_price: float,
        specialist_price: float | None,
        neural_price: float,
    ) -> NotificationReceipt:
        norm_url = self._normalize_url(opportunity.url)
        estimate = self.estimate_value(frontier_price, specialist_price, neural_price)
        landed = opportunity.landed_cost
        discount_pct = (
            ((estimate.weighted_estimate - landed) / estimate.weighted_estimate) * 100.0
            if estimate.weighted_estimate > 0
            else 0.0
        )
        discount_pct = round(discount_pct, 2)

        if norm_url in self.seen_urls:
            return NotificationReceipt(
                url=opportunity.url,
                notified=False,
                reason="duplicate_seen_url",
                landed_cost=landed,
                weighted_estimate=estimate.weighted_estimate,
                discount_pct=discount_pct,
            )

        if estimate.disagreement_pct > self.max_model_disagreement_pct:
            return NotificationReceipt(
                url=opportunity.url,
                notified=False,
                reason="high_model_disagreement",
                landed_cost=landed,
                weighted_estimate=estimate.weighted_estimate,
                discount_pct=discount_pct,
            )

        if discount_pct < self.min_discount_pct:
            return NotificationReceipt(
                url=opportunity.url,
                notified=False,
                reason="below_discount_threshold",
                landed_cost=landed,
                weighted_estimate=estimate.weighted_estimate,
                discount_pct=discount_pct,
            )

        self.seen_urls.add(norm_url)
        return NotificationReceipt(
            url=opportunity.url,
            notified=True,
            reason="alert_dispatched",
            landed_cost=landed,
            weighted_estimate=estimate.weighted_estimate,
            discount_pct=discount_pct,
        )
