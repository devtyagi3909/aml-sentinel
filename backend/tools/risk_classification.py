"""
Risk Classification Tool — Converts anomaly scores to actionable risk categories.

Maps continuous anomaly scores to discrete risk levels and determines
appropriate escalation actions based on risk level and contextual factors.
"""

import pandas as pd
import numpy as np
import time
import base64
import io
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import RISK_LOW_UPPER, RISK_MEDIUM_UPPER, RISK_HIGH_UPPER
from models.enums import RiskLevel, EscalationAction


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='#111827', edgecolor='none')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def classify_risk(
    scores_df: pd.DataFrame,
    features_df: Optional[pd.DataFrame] = None,
    customers_df: Optional[pd.DataFrame] = None,
    score_column: str = "anomaly_score",
    top_n: Optional[int] = None,
) -> dict:
    """
    Classify customers into risk categories based on anomaly scores.
    
    Args:
        scores_df: DataFrame with anomaly scores (from anomaly detection)
        features_df: Feature DataFrame for additional context
        customers_df: Customer data for enrichment
        score_column: Column name containing the anomaly score
        top_n: Only classify top N most suspicious
        
    Returns:
        dict with 'classified_df', 'summary', 'risk_distribution', 'charts'
    """
    start_time = time.time()
    charts = []
    
    df = scores_df.copy()
    
    if score_column not in df.columns:
        return {
            'classified_df': pd.DataFrame(),
            'summary': f'Score column "{score_column}" not found.',
            'risk_distribution': {},
            'charts': [],
            'duration_ms': (time.time() - start_time) * 1000
        }
    
    # Sort by score descending
    df = df.sort_values(score_column, ascending=False)
    
    if top_n:
        df = df.head(top_n)
    
    # ── Classify Risk Levels ──
    def assign_risk_level(score):
        if score <= RISK_LOW_UPPER:
            return RiskLevel.LOW
        elif score <= RISK_MEDIUM_UPPER:
            return RiskLevel.MEDIUM
        elif score <= RISK_HIGH_UPPER:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    df['risk_level'] = df[score_column].apply(assign_risk_level)
    
    # ── Assign Escalation Actions ──
    action_map = {
        RiskLevel.LOW: EscalationAction.NO_ACTION,
        RiskLevel.MEDIUM: EscalationAction.MONITOR,
        RiskLevel.HIGH: EscalationAction.REVIEW,
        RiskLevel.CRITICAL: EscalationAction.REPORT,
    }
    df['escalation_action'] = df['risk_level'].map(action_map)
    
    # ── Contextual Adjustments ──
    # Upgrade risk for PEPs
    if features_df is not None and 'pep_flag' in features_df.columns:
        pep_customers = features_df[features_df['pep_flag'] == 1].index
        pep_mask = df.index.isin(pep_customers)
        
        # Upgrade MEDIUM → HIGH for PEPs
        medium_pep = pep_mask & (df['risk_level'] == RiskLevel.MEDIUM)
        df.loc[medium_pep, 'risk_level'] = RiskLevel.HIGH
        df.loc[medium_pep, 'escalation_action'] = EscalationAction.REVIEW
        
        # Upgrade HIGH → CRITICAL for PEPs with high scores
        high_pep = pep_mask & (df['risk_level'] == RiskLevel.HIGH) & (df[score_column] > 0.65)
        df.loc[high_pep, 'risk_level'] = RiskLevel.CRITICAL
        df.loc[high_pep, 'escalation_action'] = EscalationAction.REPORT
    
    # Upgrade risk for expired KYC
    if features_df is not None and 'kyc_status_encoded' in features_df.columns:
        expired_kyc = features_df[features_df['kyc_status_encoded'] == 2].index
        expired_mask = df.index.isin(expired_kyc)
        low_expired = expired_mask & (df['risk_level'] == RiskLevel.LOW) & (df[score_column] > 0.15)
        df.loc[low_expired, 'risk_level'] = RiskLevel.MEDIUM
        df.loc[low_expired, 'escalation_action'] = EscalationAction.MONITOR
    
    # ── Enrich with customer info ──
    if customers_df is not None:
        cust = customers_df.set_index('customer_id') if 'customer_id' in customers_df.columns else customers_df
        for col in ['name', 'account_type', 'country', 'pep_flag']:
            if col in cust.columns and col not in df.columns:
                df = df.join(cust[[col]], how='left')
    
    # ── Compute Risk Distribution ──
    risk_dist = df['risk_level'].value_counts().to_dict()
    risk_distribution = {
        'critical': int(risk_dist.get(RiskLevel.CRITICAL, 0)),
        'high': int(risk_dist.get(RiskLevel.HIGH, 0)),
        'medium': int(risk_dist.get(RiskLevel.MEDIUM, 0)),
        'low': int(risk_dist.get(RiskLevel.LOW, 0)),
    }
    
    # ── Action Distribution ──
    action_dist = df['escalation_action'].value_counts().to_dict()
    action_distribution = {
        'file_sar': int(action_dist.get(EscalationAction.REPORT, 0)),
        'flag_for_review': int(action_dist.get(EscalationAction.REVIEW, 0)),
        'monitor': int(action_dist.get(EscalationAction.MONITOR, 0)),
        'no_action': int(action_dist.get(EscalationAction.NO_ACTION, 0)),
    }
    
    # ── Generate Charts ──
    
    # 1. Risk Distribution Donut Chart
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#111827')
    
    risk_colors = {
        'Critical': '#dc2626',
        'High': '#f97316',
        'Medium': '#f59e0b',
        'Low': '#22c55e'
    }
    
    labels = ['Critical', 'High', 'Medium', 'Low']
    values = [risk_distribution['critical'], risk_distribution['high'],
              risk_distribution['medium'], risk_distribution['low']]
    colors = [risk_colors[l] for l in labels]
    
    # Filter out zeros
    filtered = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    if filtered:
        f_labels, f_values, f_colors = zip(*filtered)
        
        wedges, texts, autotexts = ax.pie(
            f_values, labels=f_labels, autopct='%1.1f%%',
            colors=f_colors, textprops={'color': 'white', 'fontsize': 12},
            pctdistance=0.82, startangle=90,
            wedgeprops={'width': 0.4, 'edgecolor': '#111827', 'linewidth': 2}
        )
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        # Center text
        ax.text(0, 0, f'{len(df)}\nTotal', ha='center', va='center',
                color='white', fontsize=16, fontweight='bold')
    
    ax.set_title('Risk Distribution', color='white', fontsize=14, fontweight='bold', pad=15)
    charts.append(_fig_to_base64(fig))
    
    # 2. Top 15 Riskiest Customers Bar Chart
    top_risky = df.nlargest(15, score_column)
    if len(top_risky) > 0:
        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#1a1f35')
        
        risk_level_colors = {
            RiskLevel.CRITICAL: '#dc2626',
            RiskLevel.HIGH: '#f97316',
            RiskLevel.MEDIUM: '#f59e0b',
            RiskLevel.LOW: '#22c55e',
        }
        
        bar_colors = [risk_level_colors.get(r, '#3b82f6') for r in top_risky['risk_level']]
        
        bars = ax.barh(
            range(len(top_risky)),
            top_risky[score_column].values,
            color=bar_colors, alpha=0.9
        )
        
        ax.set_yticks(range(len(top_risky)))
        ax.set_yticklabels(top_risky.index[::-1] if top_risky.index.name == 'customer_id'
                           else [f'C-{i}' for i in range(len(top_risky))][::-1],
                           color='#d1d5db', fontsize=10)
        ax.set_xlabel('Anomaly Score', color='#9ca3af', fontsize=12)
        ax.set_title('Top 15 Riskiest Customers', color='white', fontsize=14, fontweight='bold', pad=12)
        ax.tick_params(colors='#9ca3af')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#374151')
        ax.spines['left'].set_color('#374151')
        ax.invert_yaxis()
        
        # Score labels
        for bar, score in zip(bars, top_risky[score_column].values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{score:.3f}', va='center', color='#d1d5db', fontsize=9, fontweight='bold')
        
        charts.append(_fig_to_base64(fig))
    
    # 3. Score Distribution Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#1a1f35')
    
    # Color bins by risk level
    bins = np.linspace(0, 1, 30)
    n, bins_out, patches = ax.hist(df[score_column], bins=bins, alpha=0.85, edgecolor='#111827')
    
    for patch, left_edge in zip(patches, bins_out[:-1]):
        if left_edge >= RISK_HIGH_UPPER:
            patch.set_facecolor('#dc2626')
        elif left_edge >= RISK_MEDIUM_UPPER:
            patch.set_facecolor('#f97316')
        elif left_edge >= RISK_LOW_UPPER:
            patch.set_facecolor('#f59e0b')
        else:
            patch.set_facecolor('#22c55e')
    
    # Threshold lines
    for thresh, label, color in [
        (RISK_LOW_UPPER, 'Medium', '#f59e0b'),
        (RISK_MEDIUM_UPPER, 'High', '#f97316'),
        (RISK_HIGH_UPPER, 'Critical', '#dc2626')
    ]:
        ax.axvline(x=thresh, color=color, linestyle='--', linewidth=1.5, alpha=0.7, label=f'{label} ({thresh})')
    
    ax.set_xlabel('Anomaly Score', color='#9ca3af')
    ax.set_ylabel('Frequency', color='#9ca3af')
    ax.set_title('Anomaly Score Distribution', color='white', fontsize=14, fontweight='bold', pad=12)
    ax.tick_params(colors='#9ca3af')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#374151')
    ax.spines['left'].set_color('#374151')
    ax.legend(facecolor='#1a1f35', edgecolor='#374151', labelcolor='white', fontsize=9)
    
    charts.append(_fig_to_base64(fig))
    
    duration_ms = (time.time() - start_time) * 1000
    
    summary_lines = [
        f"📊 Risk classification completed in {duration_ms:.0f}ms",
        f"👥 Classified {len(df)} customers:",
        f"   🔴 Critical (file SAR): {risk_distribution['critical']}",
        f"   🟠 High (flag for review): {risk_distribution['high']}",
        f"   🟡 Medium (monitor): {risk_distribution['medium']}",
        f"   🟢 Low (no action): {risk_distribution['low']}",
        f"",
        f"📋 Escalation Actions:",
        f"   📝 File SAR: {action_distribution['file_sar']}",
        f"   🔍 Flag for Review: {action_distribution['flag_for_review']}",
        f"   👁️ Monitor: {action_distribution['monitor']}",
    ]
    
    return {
        'classified_df': df,
        'summary': "\n".join(summary_lines),
        'risk_distribution': risk_distribution,
        'action_distribution': action_distribution,
        'charts': charts,
        'duration_ms': duration_ms
    }
