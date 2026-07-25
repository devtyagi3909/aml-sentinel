"""
Agent State — Shared state for the LangGraph agent.

Defines the TypedDict that flows through the agent graph,
tracking query parsing, execution plan, tool results, and final output.
"""

from __future__ import annotations
from typing import TypedDict, Optional, Annotated
import pandas as pd
import operator


class AgentState(TypedDict, total=False):
    """State that flows through the LangGraph agent."""
    
    # ── Input ──
    query: str
    
    # ── Parsed Query ──
    intent: str
    filters: dict
    target_entities: list[str]
    target_patterns: list[str]
    
    # ── Execution Plan ──
    execution_plan: list[str]
    current_step: int
    
    # ── Tool Results ──
    eda_results: Optional[dict]
    features_df: Optional[object]  # pd.DataFrame stored as object
    anomaly_results: Optional[dict]
    risk_results: Optional[dict]
    explanation_results: Optional[dict]
    
    # ── Aggregated Output ──
    charts: list[str]
    flagged_entities: list[dict]
    summary: str
    statistics: dict
    execution_trace: list[dict]
    
    # ── Data References ──
    transactions_df: Optional[object]
    customers_df: Optional[object]
    filtered_transactions: Optional[object]
    
    # ── Metadata ──
    error: Optional[str]
    total_analyzed: int
    total_flagged: int
    processing_time_ms: float
