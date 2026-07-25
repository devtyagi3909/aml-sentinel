"""
Pydantic schemas for AML Sentinel API requests and responses.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from .enums import RiskLevel, EscalationAction, QueryIntent, AMLPattern


class QueryRequest(BaseModel):
    """User query request."""
    query: str = Field(..., description="Natural language query for the AML agent")
    dataset_id: Optional[str] = Field(None, description="Optional dataset identifier")


class ParsedQuery(BaseModel):
    """Structured representation of a parsed user query."""
    original_query: str
    intent: QueryIntent
    filters: QueryFilters = Field(default_factory=lambda: QueryFilters())
    target_entities: list[str] = Field(default_factory=list)
    target_patterns: list[AMLPattern] = Field(default_factory=list)
    requires_eda: bool = False
    requires_feature_engineering: bool = False
    requires_anomaly_detection: bool = False
    requires_risk_classification: bool = False
    requires_explanation: bool = True


class QueryFilters(BaseModel):
    """Filters extracted from user query."""
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    customer_ids: list[str] = Field(default_factory=list)
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    transaction_types: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    channels: list[str] = Field(default_factory=list)
    min_transaction_count: Optional[int] = None


class FlaggedEntity(BaseModel):
    """A flagged transaction or customer."""
    entity_id: str
    entity_type: str = "customer"  # "customer" or "transaction"
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    escalation_action: EscalationAction
    explanation: str
    detected_patterns: list[str] = Field(default_factory=list)
    key_evidence: list[str] = Field(default_factory=list)


class ToolExecutionStep(BaseModel):
    """Record of a single tool execution."""
    tool_name: str
    status: str = "pending"  # pending, running, completed, skipped
    duration_ms: Optional[float] = None
    summary: Optional[str] = None
    skipped_reason: Optional[str] = None


class AgentResponse(BaseModel):
    """Complete agent response to a query."""
    query: str
    intent_detected: str
    filters_applied: dict = Field(default_factory=dict)
    execution_plan: list[str] = Field(default_factory=list)
    execution_trace: list[ToolExecutionStep] = Field(default_factory=list)
    flagged_entities: list[FlaggedEntity] = Field(default_factory=list)
    summary: str = ""
    eda_summary: Optional[str] = None
    charts: list[str] = Field(default_factory=list)
    statistics: dict = Field(default_factory=dict)
    total_analyzed: int = 0
    total_flagged: int = 0
    processing_time_ms: float = 0.0


class DatasetInfo(BaseModel):
    """Dataset metadata."""
    name: str
    rows: int
    columns: int
    column_names: list[str]
    date_range: Optional[str] = None
    unique_customers: int = 0
    total_transactions: int = 0
