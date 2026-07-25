"""
LangGraph Agent Orchestrator — The brain of AML Sentinel.

Coordinates the entire analysis pipeline using a dynamic state graph.
Parses queries, builds execution plans, routes through tools selectively,
and aggregates results into structured responses.

This is NOT a fixed pipeline — the agent dynamically decides which tools
to invoke based on the user's query intent.
"""

import time
import pandas as pd
import numpy as np
from typing import Optional
import asyncio

# Agent components
from agent.query_parser import parse_query, parse_query_with_llm
from agent.planner import build_execution_plan, get_plan_description
from agent.state import AgentState

# Tools
from tools.eda_tool import run_eda
from tools.feature_engineering import engineer_features
from tools.anomaly_detection import detect_anomalies
from tools.risk_classification import classify_risk
from tools.explanation_engine import generate_explanations

# Models
from models.enums import ToolName, QueryIntent
from models.schemas import (
    AgentResponse, ToolExecutionStep, FlaggedEntity,
    ParsedQuery, QueryFilters
)


class AMLSentinelAgent:
    """
    The AML Sentinel Agent — an autonomous AML analysis system.
    
    Dynamically orchestrates specialized tools based on user queries,
    providing explainable risk assessments with actionable recommendations.
    """
    
    def __init__(self, llm=None):
        """
        Initialize the agent.
        
        Args:
            llm: Optional LangChain LLM instance for enhanced query parsing
        """
        self.llm = llm
        self._transactions = None
        self._customers = None
    
    def load_data(self, transactions: pd.DataFrame, customers: pd.DataFrame = None):
        """Load transaction and customer data."""
        self._transactions = transactions.copy()
        self._customers = customers.copy() if customers is not None else None
        
        # Ensure timestamp is datetime
        if 'timestamp' in self._transactions.columns:
            self._transactions['timestamp'] = pd.to_datetime(
                self._transactions['timestamp'], errors='coerce'
            )
    
    async def process_query(self, query: str) -> AgentResponse:
        """
        Process a natural language query through the agent pipeline.
        
        This is the main entry point. The agent will:
        1. Parse the query to extract intent and filters
        2. Build a dynamic execution plan
        3. Execute only the necessary tools
        4. Aggregate results into a structured response
        """
        start_time = time.time()
        
        if self._transactions is None:
            return AgentResponse(
                query=query,
                intent_detected="error",
                summary="❌ No data loaded. Please load transaction data first.",
                processing_time_ms=0
            )
        
        # ── Step 1: Parse Query ──
        if self.llm:
            parsed = await parse_query_with_llm(query, self.llm)
        else:
            parsed = parse_query(query)
        
        # ── Step 2: Build Execution Plan ──
        plan = build_execution_plan(parsed)
        plan_description = get_plan_description(plan, parsed)
        
        # ── Step 3: Apply Data Filters ──
        filtered_txns = self._apply_filters(self._transactions, parsed)
        
        # ── Step 4: Execute Plan ──
        execution_trace = []
        charts = []
        eda_summary = None
        features_df = None
        anomaly_results = None
        risk_results = None
        explanation_results = None
        statistics = {}
        
        for tool_name in plan:
            step = ToolExecutionStep(tool_name=tool_name, status="running")
            step_start = time.time()
            
            try:
                if tool_name == ToolName.EDA.value:
                    result = self._run_eda(filtered_txns, parsed)
                    eda_summary = result.get('summary', '')
                    charts.extend(result.get('charts', []))
                    statistics.update(result.get('statistics', {}))
                    step.summary = f"Analyzed {len(filtered_txns):,} transactions, generated {len(result.get('charts', []))} charts"
                
                elif tool_name == ToolName.FEATURE_ENGINEERING.value:
                    result = self._run_feature_engineering(filtered_txns, parsed)
                    features_df = result.get('features_df')
                    step.summary = (
                        f"Generated {len(result.get('feature_names', []))} features "
                        f"for {len(features_df) if features_df is not None else 0} customers"
                    )
                
                elif tool_name == ToolName.ANOMALY_DETECTION.value:
                    if features_df is None or len(features_df) == 0:
                        step.status = "skipped"
                        step.skipped_reason = "No features available"
                        step.duration_ms = 0
                        execution_trace.append(step)
                        continue
                    
                    result = self._run_anomaly_detection(features_df, filtered_txns, parsed)
                    anomaly_results = result
                    step.summary = (
                        f"Detected {result.get('anomaly_count', 0)} anomalies "
                        f"({result.get('high_risk_count', 0)} high risk, "
                        f"{result.get('critical_count', 0)} critical)"
                    )
                
                elif tool_name == ToolName.RISK_CLASSIFICATION.value:
                    if anomaly_results is None or 'scores_df' not in anomaly_results:
                        step.status = "skipped"
                        step.skipped_reason = "No anomaly scores available"
                        step.duration_ms = 0
                        execution_trace.append(step)
                        continue
                    
                    result = self._run_risk_classification(
                        anomaly_results['scores_df'], features_df
                    )
                    risk_results = result
                    charts.extend(result.get('charts', []))
                    step.summary = (
                        f"Classified {sum(result.get('risk_distribution', {}).values())} entities: "
                        f"{result['risk_distribution'].get('critical', 0)} critical, "
                        f"{result['risk_distribution'].get('high', 0)} high"
                    )
                
                elif tool_name == ToolName.EXPLANATION.value:
                    if risk_results is None or features_df is None:
                        step.status = "skipped"
                        step.skipped_reason = "No classified entities to explain"
                        step.duration_ms = 0
                        execution_trace.append(step)
                        continue
                    
                    result = self._run_explanations(
                        risk_results['classified_df'], features_df, 
                        filtered_txns, parsed
                    )
                    explanation_results = result
                    step.summary = f"Generated explanations for {len(result.get('explanations', []))} entities"
                
                step.status = "completed"
                step.duration_ms = (time.time() - step_start) * 1000
                
            except Exception as e:
                step.status = "error"
                step.summary = f"Error: {str(e)}"
                step.duration_ms = (time.time() - step_start) * 1000
                print(f"Tool {tool_name} error: {e}")
                import traceback
                traceback.print_exc()
            
            execution_trace.append(step)
        
        # ── Step 5: Build Response ──
        flagged_entities = []
        if explanation_results and 'explanations' in explanation_results:
            for exp in explanation_results['explanations']:
                flagged_entities.append(FlaggedEntity(
                    entity_id=exp['entity_id'],
                    entity_type=exp.get('entity_type', 'customer'),
                    risk_score=exp['risk_score'],
                    risk_level=exp['risk_level'],
                    escalation_action=exp['escalation_action'],
                    explanation=exp['explanation'],
                    detected_patterns=exp.get('detected_patterns', []),
                    key_evidence=exp.get('key_evidence', []),
                ))
        
        # Build overall summary
        summary_parts = [f"🔍 Query: \"{query}\"", "", plan_description, ""]
        
        if eda_summary:
            summary_parts.append("━━━ EDA Results ━━━")
            summary_parts.append(eda_summary)
            summary_parts.append("")
        
        if anomaly_results:
            summary_parts.append("━━━ Anomaly Detection ━━━")
            summary_parts.append(anomaly_results.get('summary', ''))
            summary_parts.append("")
        
        if risk_results:
            summary_parts.append("━━━ Risk Classification ━━━")
            summary_parts.append(risk_results.get('summary', ''))
            summary_parts.append("")
        
        if explanation_results:
            summary_parts.append("━━━ Explanations ━━━")
            summary_parts.append(explanation_results.get('summary', ''))
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Handle threshold queries specially
        if parsed.intent == QueryIntent.THRESHOLD_QUERY and features_df is not None:
            threshold_result = self._handle_threshold_query(
                filtered_txns, features_df, parsed
            )
            if threshold_result:
                summary_parts.append("")
                summary_parts.append("━━━ Query Results ━━━")
                summary_parts.append(threshold_result['summary'])
                statistics.update(threshold_result.get('statistics', {}))
        
        # Handle aggregation queries
        if parsed.intent == QueryIntent.AGGREGATION:
            agg_result = self._handle_aggregation_query(filtered_txns, parsed)
            if agg_result:
                summary_parts.append("")
                summary_parts.append(agg_result['summary'])
                statistics.update(agg_result.get('statistics', {}))
        
        return AgentResponse(
            query=query,
            intent_detected=parsed.intent.value,
            filters_applied=parsed.filters.model_dump(exclude_none=True),
            execution_plan=plan,
            execution_trace=execution_trace,
            flagged_entities=flagged_entities,
            summary="\n".join(summary_parts),
            eda_summary=eda_summary,
            charts=charts,
            statistics=statistics,
            total_analyzed=len(filtered_txns),
            total_flagged=len(flagged_entities),
            processing_time_ms=processing_time_ms,
        )
    
    def _apply_filters(self, df: pd.DataFrame, parsed: ParsedQuery) -> pd.DataFrame:
        """Apply parsed query filters to transaction data."""
        filtered = df.copy()
        f = parsed.filters
        
        if f.customer_ids:
            filtered = filtered[filtered['customer_id'].isin(f.customer_ids)]
        
        if f.date_start and 'timestamp' in filtered.columns:
            filtered = filtered[filtered['timestamp'] >= pd.to_datetime(f.date_start)]
        
        if f.date_end and 'timestamp' in filtered.columns:
            filtered = filtered[filtered['timestamp'] <= pd.to_datetime(f.date_end)]
        
        if f.min_amount is not None and 'amount' in filtered.columns:
            filtered = filtered[filtered['amount'] >= f.min_amount]
        
        if f.max_amount is not None and 'amount' in filtered.columns:
            filtered = filtered[filtered['amount'] <= f.max_amount]
        
        if f.transaction_types and 'transaction_type' in filtered.columns:
            filtered = filtered[filtered['transaction_type'].isin(f.transaction_types)]
        
        if f.countries and 'counterparty_country' in filtered.columns:
            filtered = filtered[filtered['counterparty_country'].isin(f.countries)]
        
        return filtered
    
    def _run_eda(self, transactions: pd.DataFrame, parsed: ParsedQuery) -> dict:
        """Run EDA tool with appropriate parameters."""
        date_range = None
        if parsed.filters.date_start or parsed.filters.date_end:
            date_range = (parsed.filters.date_start, parsed.filters.date_end)
        
        return run_eda(
            transactions=transactions,
            customers=self._customers,
            subset_customer_ids=parsed.filters.customer_ids or None,
            date_range=date_range,
        )
    
    def _run_feature_engineering(self, transactions: pd.DataFrame, 
                                  parsed: ParsedQuery) -> dict:
        """Run feature engineering with appropriate parameters."""
        # Determine feature focus based on target patterns
        pattern_focus = None
        if parsed.target_patterns:
            pattern_focus = parsed.target_patterns[0].value if hasattr(parsed.target_patterns[0], 'value') else str(parsed.target_patterns[0])
        
        return engineer_features(
            transactions=transactions,
            customers=self._customers,
            customer_ids=parsed.filters.customer_ids or None,
            feature_set="all",
            pattern_focus=pattern_focus,
        )
    
    def _run_anomaly_detection(self, features_df: pd.DataFrame,
                                 transactions: pd.DataFrame,
                                 parsed: ParsedQuery) -> dict:
        """Run anomaly detection with appropriate parameters."""
        return detect_anomalies(
            features_df=features_df,
            transactions=transactions,
            method="ensemble",
            pattern_focus=parsed.target_patterns[0].value if parsed.target_patterns else None,
            customer_ids=parsed.filters.customer_ids or None,
        )
    
    def _run_risk_classification(self, scores_df: pd.DataFrame,
                                   features_df: pd.DataFrame) -> dict:
        """Run risk classification."""
        return classify_risk(
            scores_df=scores_df,
            features_df=features_df,
            customers_df=self._customers,
        )
    
    def _run_explanations(self, classified_df: pd.DataFrame,
                           features_df: pd.DataFrame,
                           transactions: pd.DataFrame,
                           parsed: ParsedQuery) -> dict:
        """Generate explanations for flagged entities."""
        # For entity lookups, explain all; otherwise top 20
        top_n = None if parsed.intent == QueryIntent.ENTITY_LOOKUP else 20
        
        return generate_explanations(
            classified_df=classified_df,
            features_df=features_df,
            transactions=transactions,
            customers_df=self._customers,
            top_n=top_n,
            query_context=parsed.original_query,
        )
    
    def _handle_threshold_query(self, transactions: pd.DataFrame,
                                  features_df: pd.DataFrame,
                                  parsed: ParsedQuery) -> dict:
        """Handle threshold/count-based queries directly."""
        results = {}
        
        if parsed.filters.min_transaction_count and 'amount' in transactions.columns:
            threshold = parsed.filters.min_transaction_count
            
            # Filter based on amount constraints
            txns = transactions.copy()
            if parsed.filters.max_amount:
                txns = txns[txns['amount'] < parsed.filters.max_amount]
            if parsed.filters.min_amount:
                txns = txns[txns['amount'] >= parsed.filters.min_amount]
            
            # Check for cash-specific queries
            if 'cash' in parsed.original_query.lower() and 'is_cash' in txns.columns:
                txns = txns[txns['is_cash'] == True]
            
            # Count per customer
            counts = txns.groupby('customer_id').size()
            qualifying = counts[counts >= threshold]
            
            summary_lines = [
                f"📊 Found {len(qualifying)} customers with {threshold}+ matching transactions",
                f"Top customers by count:"
            ]
            
            for cid, count in qualifying.nlargest(10).items():
                total = txns[txns['customer_id'] == cid]['amount'].sum()
                summary_lines.append(f"   • {cid}: {count} transactions (${total:,.0f} total)")
            
            results['summary'] = "\n".join(summary_lines)
            results['statistics'] = {
                'qualifying_customers': int(len(qualifying)),
                'threshold': threshold,
            }
        
        return results if results else None
    
    def _handle_aggregation_query(self, transactions: pd.DataFrame,
                                    parsed: ParsedQuery) -> dict:
        """Handle aggregation queries (average, total, etc.)."""
        q = parsed.original_query.lower()
        
        if 'amount' in transactions.columns:
            stats = transactions['amount'].describe().to_dict()
            summary = f"📊 Transaction Statistics:\n"
            summary += f"   • Count: {stats['count']:,.0f}\n"
            summary += f"   • Mean: ${stats['mean']:,.2f}\n"
            summary += f"   • Median: ${stats['50%']:,.2f}\n"
            summary += f"   • Std Dev: ${stats['std']:,.2f}\n"
            summary += f"   • Min: ${stats['min']:,.2f}\n"
            summary += f"   • Max: ${stats['max']:,.2f}"
            
            return {
                'summary': summary,
                'statistics': {k: round(v, 2) for k, v in stats.items()}
            }
        
        return None
