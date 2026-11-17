from __future__ import annotations

from pydantic import BaseModel, Field


class DealOpportunity(BaseModel):
    title: str
    url: str
    listed_price: float = Field(ge=0.0)
    shipping_cost: float = Field(default=0.0, ge=0.0)
    estimated_tax: float = Field(default=0.0, ge=0.0)

    @property
    def landed_cost(self) -> float:
        return round(self.listed_price + self.shipping_cost + self.estimated_tax, 2)


class EnsembleEstimate(BaseModel):
    frontier_price: float = Field(gt=0.0)
    specialist_price: float | None = Field(default=None, gt=0.0)
    neural_price: float = Field(gt=0.0)
    weighted_estimate: float = Field(gt=0.0)
    disagreement_pct: float = Field(ge=0.0)
    specialist_fallback: bool = False


class NotificationReceipt(BaseModel):
    url: str
    notified: bool
    reason: str
    landed_cost: float
    weighted_estimate: float
    discount_pct: float
