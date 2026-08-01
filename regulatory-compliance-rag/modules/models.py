from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Decision = Literal["APPROVED", "BLOCKED", "NEEDS_REVIEW"]


class StrategyCheckRequest(BaseModel):
	strategy_name: str | None = Field(default=None, description="Optional strategy label")
	strategy: str = Field(..., min_length=5, description="Proposed trading strategy")
	description: str | None = Field(default=None, description="Detailed strategy description")
	asset_class: str | None = None
	leverage_ratio: float | None = Field(default=None, ge=0)
	short_selling: bool | None = None
	use_insider_info: bool | None = None
	geographic_scope: list[str] = Field(default_factory=list)
	holding_period_days: int | None = Field(default=None, ge=0)



class Citation(BaseModel):
    source_id: str
    source_name: str
    regulation_title: str
    jurisdiction: str
    effective_date: str
    section: str
    chunk_id: str
    confidence: float
    retrieval_score: float | None = None
    excerpt: str


class StrategyCheckResponse(BaseModel):
	strategy_id: str
	decision: Decision
	confidence: float
	reason: str
	citations: list[Citation]
	retrieved_count: int
	risk_flags: list[str] = Field(default_factory=list)
	recommendations: list[str] = Field(default_factory=list)
	timestamp: str


class RegulationInfo(BaseModel):
	source_id: str
	source_name: str
	regulation_title: str
	jurisdiction: str
	effective_date: str
	chunk_count: int
	text_preview: str


class StrategyAuditRecord(BaseModel):
	strategy_id: str
	timestamp: str
	decision: Decision
	confidence: float
	strategy: str
	retrieved_count: int
	reason: str


class UploadRegulationResponse(BaseModel):
	source_id: str
	source_name: str
	regulation_title: str
	jurisdiction: str
	effective_date: str
	chunk_count: int


class RegulationSearchResponse(BaseModel):
	query: str
	results: list[Citation]
	retrieval_time_ms: int


class ErrorResponse(BaseModel):
	detail: str

