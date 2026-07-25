"""
Explanation Engine — Generates human-readable explanations for AML flags.

Creates natural language explanations for why a transaction or customer
was flagged as suspicious, referencing specific patterns, transactions,
and risk factors. Uses LLM for synthesis when available, falls back to
template-based explanations.
"""

import pandas as pd
import numpy as np
import time
from typing import Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    CTR_THRESHOLD, STRUCTURING_LOWER, STRUCTURING_COUNT_THRESHOLD,
    HIGH_RISK_COUNTRIES
)
from models.enums import RiskLevel, EscalationAction


def _get_pattern_explanation(customer_id: str, features: pd.Series, 
                              transactions: Optional[pd.DataFrame] = None) -> list:
    """Generate pattern-specific evidence descriptions."""
    evidence = []
    
    # Structuring evidence
    near_ctr = features.get('cash_near_ctr_count', 0)
    if near_ctr >= STRUCTURING_COUNT_THRESHOLD:
        cluster = features.get('structuring_cluster_max', near_ctr)
        total = features.get('cash_near_ctr_total', 0)
        gap = features.get('near_ctr_avg_gap_hours', 0)
        evidence.append(
            f"STRUCTURING DETECTED: {int(near_ctr)} cash deposits in the ${STRUCTURING_LOWER:,}–$9,999 range "
            f"(total: ${total:,.0f}), with up to {int(cluster)} occurring within a 7-day window"
            + (f", averaging {gap:.0f} hours apart" if gap > 0 and gap < 500 else "")
            + f". This is a classic structuring pattern to evade the ${CTR_THRESHOLD:,} CTR reporting threshold."
        )
    
    # Rapid movement evidence
    rapid = features.get('rapid_movement_count', 0)
    if rapid > 0:
        evidence.append(
            f"RAPID MOVEMENT: {int(rapid)} instance(s) of large deposits (>$20K) followed by "
            f"outbound transfers within 24 hours. This layering pattern is designed to quickly "
            f"move funds through the account before detection."
        )
    
    # Velocity spike evidence
    velocity = features.get('velocity_ratio_7d', 0)
    if velocity >= 5:
        txn_7d = features.get('txn_count_7d', 0)
        evidence.append(
            f"VELOCITY SPIKE: Transaction frequency in the last 7 days is {velocity:.1f}x "
            f"the customer's historical average ({int(txn_7d)} transactions in 7 days). "
            f"Sudden increases in activity warrant investigation."
        )
    
    # Geographic risk evidence
    geo_pct = features.get('high_risk_country_pct', 0)
    if geo_pct > 0.2:
        hr_txns = features.get('high_risk_country_txns', 0)
        hr_vol = features.get('high_risk_country_volume', 0)
        evidence.append(
            f"GEOGRAPHIC RISK: {geo_pct:.0%} of transactions ({int(hr_txns)} total, "
            f"${hr_vol:,.0f} volume) involve high-risk jurisdictions (FATF grey/black list). "
            f"Elevated counterparty risk requires enhanced due diligence."
        )
    
    # Dormancy activation evidence
    dormancy = features.get('dormancy_flag', 0)
    if dormancy == 1:
        evidence.append(
            f"DORMANT ACCOUNT ACTIVATION: This account had a period of 90+ days of "
            f"inactivity followed by sudden transaction activity. Reactivation of dormant "
            f"accounts is a known AML red flag, especially with high-value transactions."
        )
    
    # Unusual timing
    unusual_time = features.get('unusual_time_pct', 0)
    if unusual_time > 0.4:
        evidence.append(
            f"UNUSUAL TIMING: {unusual_time:.0%} of transactions occur during "
            f"off-hours (nights/weekends), which is {unusual_time/0.28:.1f}x "
            f"the expected rate for typical banking activity."
        )
    
    # PEP status
    pep = features.get('pep_flag', 0)
    if pep == 1:
        evidence.append(
            f"POLITICALLY EXPOSED PERSON: Customer is flagged as a PEP, requiring "
            f"enhanced due diligence per FATF Recommendation 12."
        )
    
    # Amount volatility
    amount_cv = features.get('amount_cv', 0)
    if amount_cv > 3:
        evidence.append(
            f"HIGH AMOUNT VOLATILITY: Coefficient of variation of transaction amounts "
            f"is {amount_cv:.1f}, indicating highly irregular transaction sizes "
            f"(normal range: 0.5–2.0)."
        )
    
    # === Graph Network Features ===
    pagerank = features.get('network_pagerank', 0)
    if pagerank > 0.05:
        evidence.append(
            f"NETWORK CENTRALITY (PAGERANK): Customer has an unusually high PageRank score "
            f"({pagerank:.4f}), indicating they act as a central hub or 'money mule' within "
            f"a broader network of wire transfers."
        )
        
    circular_flow = features.get('circular_flow_count', 0)
    if circular_flow > 0:
        evidence.append(
            f"CIRCULAR FLOW DETECTED: Mathematical graph analysis detected {int(circular_flow)} "
            f"instance(s) where this customer is part of a circular money flow "
            f"(e.g., Round-Tripping A → B → C → A). This is a highly critical laundering indicator."
        )
    
    # If no specific patterns, provide generic explanation
    if not evidence:
        score = features.get('anomaly_score', features.get('ensemble_score', 0))
        if score > 0.3:
            evidence.append(
                f"STATISTICAL ANOMALY: Customer's transaction behavior deviates "
                f"significantly from the baseline population across multiple features. "
                f"Combined anomaly score: {score:.3f}."
            )
    
    return evidence


def generate_explanations(
    classified_df: pd.DataFrame,
    features_df: pd.DataFrame,
    transactions: Optional[pd.DataFrame] = None,
    customers_df: Optional[pd.DataFrame] = None,
    top_n: Optional[int] = None,
    risk_filter: Optional[str] = None,
    query_context: Optional[str] = None,
) -> dict:
    """
    Generate natural language explanations for flagged entities.
    
    Args:
        classified_df: Risk-classified DataFrame (from risk classification)
        features_df: Feature DataFrame
        transactions: Original transaction data
        customers_df: Customer information
        top_n: Generate explanations for top N entities
        risk_filter: Only explain entities with this risk level or higher
        query_context: Original user query for context-aware explanations
        
    Returns:
        dict with 'explanations' (list of dicts), 'summary', 'duration_ms'
    """
    start_time = time.time()
    
    df = classified_df.copy()
    
    # Filter by risk level
    risk_order = [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW]
    if risk_filter:
        filter_level = RiskLevel(risk_filter) if isinstance(risk_filter, str) else risk_filter
        filter_idx = risk_order.index(filter_level)
        allowed_levels = risk_order[:filter_idx + 1]
        df = df[df['risk_level'].isin(allowed_levels)]
    else:
        # Default: only explain non-LOW risk
        df = df[df['risk_level'] != RiskLevel.LOW]
    
    # Sort by anomaly score
    score_col = 'anomaly_score' if 'anomaly_score' in df.columns else df.select_dtypes(include=[np.number]).columns[-1]
    df = df.sort_values(score_col, ascending=False)
    
    if top_n:
        df = df.head(top_n)
    
    explanations = []
    
    for customer_id in df.index:
        row = df.loc[customer_id]
        
        # Get features for this customer
        if customer_id in features_df.index:
            cust_features = features_df.loc[customer_id]
        else:
            cust_features = pd.Series(dtype=float)
        
        # Merge with classification scores and remove duplicates
        combined = pd.concat([cust_features, row])
        combined = combined[~combined.index.duplicated(keep='last')]
        
        # Get customer info
        customer_name = "Unknown"
        customer_country = "N/A"
        account_type = "N/A"
        
        if customers_df is not None:
            cust_info = customers_df[customers_df['customer_id'] == customer_id] if 'customer_id' in customers_df.columns else pd.DataFrame()
            if len(cust_info) > 0:
                customer_name = cust_info.iloc[0].get('name', 'Unknown')
                customer_country = cust_info.iloc[0].get('country', 'N/A')
                account_type = cust_info.iloc[0].get('account_type', 'N/A')
        
        # Generate evidence
        evidence = _get_pattern_explanation(customer_id, combined, transactions)
        
        # Detected patterns
        detected_patterns = []
        if combined.get('cash_near_ctr_count', 0) >= STRUCTURING_COUNT_THRESHOLD:
            detected_patterns.append('structuring')
        if combined.get('rapid_movement_count', 0) > 0:
            detected_patterns.append('rapid_movement')
        if combined.get('velocity_ratio_7d', 0) >= 5:
            detected_patterns.append('velocity_spike')
        if combined.get('high_risk_country_pct', 0) > 0.2:
            detected_patterns.append('geographic_risk')
        if combined.get('dormancy_flag', 0) == 1:
            detected_patterns.append('dormant_activation')
        
        risk_level = row.get('risk_level', RiskLevel.MEDIUM)
        escalation = row.get('escalation_action', EscalationAction.MONITOR)
        score = float(row.get(score_col, 0))
        
        # Build explanation summary
        risk_emoji = {
            RiskLevel.CRITICAL: '🔴',
            RiskLevel.HIGH: '🟠',
            RiskLevel.MEDIUM: '🟡',
            RiskLevel.LOW: '🟢'
        }
        
        action_text = {
            EscalationAction.REPORT: 'FILE SUSPICIOUS ACTIVITY REPORT (SAR)',
            EscalationAction.REVIEW: 'FLAG FOR COMPLIANCE REVIEW',
            EscalationAction.MONITOR: 'ADD TO MONITORING WATCHLIST',
            EscalationAction.NO_ACTION: 'NO ACTION REQUIRED',
        }
        
        explanation_text = (
            f"{risk_emoji.get(risk_level, '⚪')} Customer {customer_id} "
            f"({customer_name}) — {risk_level.value.upper()} RISK (score: {score:.3f})\n"
            f"Account: {account_type} | Country: {customer_country}\n\n"
            f"Evidence:\n" + "\n".join(f"  • {e}" for e in evidence) + "\n\n"
            f"Recommended Action: {action_text.get(escalation, 'REVIEW')}"
        )
        
        explanation_entry = {
            'entity_id': str(customer_id),
            'entity_type': 'customer',
            'risk_score': score,
            'risk_level': risk_level.value if isinstance(risk_level, RiskLevel) else str(risk_level),
            'escalation_action': escalation.value if isinstance(escalation, EscalationAction) else str(escalation),
            'explanation': explanation_text,
            'detected_patterns': detected_patterns,
            'key_evidence': evidence,
            'customer_name': customer_name,
            'customer_country': customer_country,
            'account_type': account_type,
        }
        
        explanations.append(explanation_entry)
    
    duration_ms = (time.time() - start_time) * 1000
    
    # Summary
    pattern_counts = {}
    for exp in explanations:
        for p in exp['detected_patterns']:
            pattern_counts[p] = pattern_counts.get(p, 0) + 1
    
    summary_lines = [
        f"📝 Generated explanations for {len(explanations)} flagged entities in {duration_ms:.0f}ms",
    ]
    
    if pattern_counts:
        summary_lines.append(f"🔍 Detected patterns across flagged entities:")
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            summary_lines.append(f"   • {pattern.replace('_', ' ').title()}: {count} customers")
    
    risk_counts = {}
    for exp in explanations:
        rl = exp['risk_level']
        risk_counts[rl] = risk_counts.get(rl, 0) + 1
    
    if risk_counts:
        summary_lines.append(f"📊 Breakdown: " + 
                           ", ".join(f"{k.upper()}: {v}" for k, v in risk_counts.items()))
    
    return {
        'explanations': explanations,
        'summary': "\n".join(summary_lines),
        'pattern_counts': pattern_counts,
        'duration_ms': duration_ms
    }
