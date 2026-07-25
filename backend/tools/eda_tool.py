"""
EDA Tool — Performs exploratory data analysis on transaction and customer data.

This tool provides dataset profiling, distribution analysis, time-series patterns,
and generates visualizations for compliance reviewers.
"""

import pandas as pd
import numpy as np
import base64
import io
import time
from typing import Optional

# Try plotly first, fall back to matplotlib
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='#111827', edgecolor='none')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def _plotly_to_base64(fig) -> str:
    """Convert a plotly figure to base64 string."""
    img_bytes = pio.to_image(fig, format='png', width=800, height=500,
                             scale=2)
    return base64.b64encode(img_bytes).decode('utf-8')


def _style_dark_chart(fig, ax, title: str):
    """Apply dark theme to matplotlib chart."""
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#1a1f35')
    ax.set_title(title, color='white', fontsize=14, fontweight='bold', pad=12)
    ax.tick_params(colors='#9ca3af', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#374151')
    ax.spines['left'].set_color('#374151')
    ax.xaxis.label.set_color('#9ca3af')
    ax.yaxis.label.set_color('#9ca3af')
    ax.grid(axis='y', alpha=0.15, color='#4b5563')


def run_eda(
    transactions: pd.DataFrame,
    customers: Optional[pd.DataFrame] = None,
    subset_customer_ids: Optional[list] = None,
    date_range: Optional[tuple] = None,
    focus_areas: Optional[list] = None,
) -> dict:
    """
    Run exploratory data analysis on transaction data.
    
    Args:
        transactions: Transaction DataFrame
        customers: Customer DataFrame (optional)
        subset_customer_ids: Analyze only these customers
        date_range: (start_date, end_date) tuple
        focus_areas: List of specific EDA aspects to focus on
        
    Returns:
        dict with 'summary' (str), 'statistics' (dict), 'charts' (list of base64)
    """
    start_time = time.time()
    charts = []
    stats = {}
    summaries = []
    
    df = transactions.copy()
    
    # Apply filters
    if subset_customer_ids:
        df = df[df['customer_id'].isin(subset_customer_ids)]
    
    if date_range and 'timestamp' in df.columns:
        if df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        if date_range[0]:
            df = df[df['timestamp'] >= pd.to_datetime(date_range[0])]
        if date_range[1]:
            df = df[df['timestamp'] <= pd.to_datetime(date_range[1])]
    
    if len(df) == 0:
        return {
            'summary': 'No transactions found matching the specified filters.',
            'statistics': {},
            'charts': [],
            'duration_ms': (time.time() - start_time) * 1000
        }
    
    # ── 1. Basic Dataset Profile ──
    stats['total_transactions'] = len(df)
    stats['unique_customers'] = df['customer_id'].nunique()
    stats['date_range'] = {
        'start': str(df['timestamp'].min()) if 'timestamp' in df.columns else 'N/A',
        'end': str(df['timestamp'].max()) if 'timestamp' in df.columns else 'N/A',
    }
    stats['missing_values'] = df.isnull().sum().to_dict()
    stats['columns'] = list(df.columns)
    
    summaries.append(f"📊 Dataset contains {len(df):,} transactions from {stats['unique_customers']:,} unique customers.")
    
    if 'timestamp' in df.columns:
        summaries.append(f"📅 Date range: {stats['date_range']['start'][:10]} to {stats['date_range']['end'][:10]}")
    
    # ── 2. Amount Distribution ──
    if 'amount' in df.columns:
        amount_stats = df['amount'].describe().to_dict()
        stats['amount_distribution'] = {k: round(v, 2) for k, v in amount_stats.items()}
        
        summaries.append(
            f"💰 Transaction amounts: mean=${amount_stats['mean']:,.2f}, "
            f"median=${amount_stats['50%']:,.2f}, max=${amount_stats['max']:,.2f}"
        )
        
        # Amount histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        _style_dark_chart(fig, ax, 'Transaction Amount Distribution')
        
        # Clip for visualization
        clip_amount = min(df['amount'].quantile(0.99), 50000)
        clipped = df['amount'].clip(upper=clip_amount)
        
        ax.hist(clipped, bins=50, color='#3b82f6', alpha=0.8, edgecolor='#1e40af')
        ax.axvline(x=10000, color='#ef4444', linestyle='--', linewidth=2, label='$10K CTR Threshold')
        ax.set_xlabel('Amount ($)')
        ax.set_ylabel('Frequency')
        ax.legend(facecolor='#1a1f35', edgecolor='#374151', labelcolor='white')
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        charts.append(_fig_to_base64(fig))
        
        # Amount box plot by transaction type
        if 'transaction_type' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            _style_dark_chart(fig, ax, 'Amount by Transaction Type')
            
            types = df['transaction_type'].unique()[:6]
            data_by_type = [df[df['transaction_type'] == t]['amount'].clip(upper=clip_amount).values for t in types]
            
            bp = ax.boxplot(data_by_type, tick_labels=types, patch_artist=True,
                           boxprops=dict(facecolor='#3b82f6', alpha=0.6),
                           medianprops=dict(color='#fbbf24', linewidth=2),
                           whiskerprops=dict(color='#9ca3af'),
                           capprops=dict(color='#9ca3af'),
                           flierprops=dict(markerfacecolor='#ef4444', markersize=3, alpha=0.3))
            ax.set_ylabel('Amount ($)')
            ax.tick_params(axis='x', rotation=30)
            charts.append(_fig_to_base64(fig))
    
    # ── 3. Transaction Type Distribution ──
    if 'transaction_type' in df.columns:
        type_counts = df['transaction_type'].value_counts()
        stats['transaction_types'] = type_counts.to_dict()
        
        fig, ax = plt.subplots(figsize=(7, 5))
        _style_dark_chart(fig, ax, 'Transaction Type Distribution')
        
        colors = ['#3b82f6', '#f59e0b', '#22c55e', '#ef4444', '#8b5cf6', '#ec4899']
        bars = ax.barh(type_counts.index[:6], type_counts.values[:6],
                       color=colors[:len(type_counts[:6])], alpha=0.85)
        ax.set_xlabel('Count')
        
        for bar, val in zip(bars, type_counts.values[:6]):
            ax.text(bar.get_width() + max(type_counts.values) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:,}', va='center', color='#d1d5db', fontsize=10)
        charts.append(_fig_to_base64(fig))
    
    # ── 4. Transaction Volume Over Time ──
    if 'timestamp' in df.columns:
        if df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        daily = df.set_index('timestamp').resample('D').agg(
            count=('amount', 'count'),
            total=('amount', 'sum')
        ).reset_index()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        fig.patch.set_facecolor('#111827')
        
        _style_dark_chart(fig, ax1, 'Daily Transaction Count')
        ax1.fill_between(daily['timestamp'], daily['count'], alpha=0.3, color='#3b82f6')
        ax1.plot(daily['timestamp'], daily['count'], color='#60a5fa', linewidth=1.5)
        ax1.set_ylabel('Count')
        
        _style_dark_chart(fig, ax2, 'Daily Transaction Volume ($)')
        ax2.fill_between(daily['timestamp'], daily['total'], alpha=0.3, color='#f59e0b')
        ax2.plot(daily['timestamp'], daily['total'], color='#fbbf24', linewidth=1.5)
        ax2.set_ylabel('Volume ($)')
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        plt.tight_layout()
        charts.append(_fig_to_base64(fig))
    
    # ── 5. Channel Distribution ──
    if 'channel' in df.columns:
        channel_counts = df['channel'].value_counts()
        stats['channels'] = channel_counts.to_dict()
        
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor('#111827')
        
        colors = ['#3b82f6', '#f59e0b', '#22c55e', '#ef4444', '#8b5cf6']
        wedges, texts, autotexts = ax.pie(
            channel_counts.values[:5], labels=channel_counts.index[:5],
            autopct='%1.1f%%', colors=colors[:len(channel_counts[:5])],
            textprops={'color': 'white', 'fontsize': 11},
            pctdistance=0.75, startangle=90
        )
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        ax.set_title('Transaction Channels', color='white', fontsize=14, fontweight='bold')
        charts.append(_fig_to_base64(fig))
    
    # ── 6. Top Customers by Transaction Volume ──
    if 'customer_id' in df.columns and 'amount' in df.columns:
        top_customers = df.groupby('customer_id').agg(
            total_amount=('amount', 'sum'),
            txn_count=('amount', 'count'),
            avg_amount=('amount', 'mean')
        ).nlargest(10, 'total_amount').reset_index()
        
        stats['top_customers'] = top_customers.to_dict(orient='records')
        
        fig, ax = plt.subplots(figsize=(8, 5))
        _style_dark_chart(fig, ax, 'Top 10 Customers by Total Volume')
        
        bars = ax.barh(top_customers['customer_id'][::-1],
                       top_customers['total_amount'][::-1],
                       color='#3b82f6', alpha=0.85)
        ax.set_xlabel('Total Volume ($)')
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        charts.append(_fig_to_base64(fig))
    
    # ── 7. Cash vs Non-Cash ──
    if 'is_cash' in df.columns:
        cash_stats = df.groupby('is_cash')['amount'].agg(['count', 'sum', 'mean'])
        stats['cash_analysis'] = {
            'cash_transactions': int(cash_stats.loc[True, 'count']) if True in cash_stats.index else 0,
            'non_cash_transactions': int(cash_stats.loc[False, 'count']) if False in cash_stats.index else 0,
            'cash_pct': round(float(df['is_cash'].mean() * 100), 1) if df['is_cash'].any() else 0,
        }
        summaries.append(f"💵 Cash transactions: {stats['cash_analysis']['cash_pct']}% of total")
    
    # ── 8. Geographic Distribution ──
    if 'counterparty_country' in df.columns:
        country_counts = df['counterparty_country'].value_counts().head(15)
        stats['top_countries'] = country_counts.to_dict()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        _style_dark_chart(fig, ax, 'Top 15 Counterparty Countries')
        
        country_colors = ['#ef4444' if c in ['KY', 'PA', 'VG', 'BS', 'BZ', 'MM', 'KP', 'IR', 'SY', 'AF']
                         else '#3b82f6' for c in country_counts.index]
        
        bars = ax.bar(country_counts.index, country_counts.values,
                      color=country_colors, alpha=0.85)
        ax.set_xlabel('Country')
        ax.set_ylabel('Transaction Count')
        ax.tick_params(axis='x', rotation=45)
        
        # Add legend for high-risk
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ef4444', alpha=0.85, label='High-Risk Jurisdiction'),
            Patch(facecolor='#3b82f6', alpha=0.85, label='Standard')
        ]
        ax.legend(handles=legend_elements, facecolor='#1a1f35', edgecolor='#374151',
                 labelcolor='white', loc='upper right')
        charts.append(_fig_to_base64(fig))
    
    # ── 9. Near-CTR Analysis (Structuring Indicator) ──
    if 'amount' in df.columns and 'is_cash' in df.columns:
        near_ctr = df[(df['amount'] >= 8000) & (df['amount'] < 10000) & (df['is_cash'] == True)]
        stats['near_ctr_count'] = len(near_ctr)
        stats['near_ctr_pct'] = round(len(near_ctr) / max(len(df[df['is_cash'] == True]), 1) * 100, 2)
        
        if len(near_ctr) > 0:
            summaries.append(
                f"⚠️ Near-CTR cash transactions ($8K-$10K): {len(near_ctr)} "
                f"({stats['near_ctr_pct']}% of cash transactions) — potential structuring indicator"
            )
            
            fig, ax = plt.subplots(figsize=(8, 5))
            _style_dark_chart(fig, ax, 'Cash Deposits Near CTR Threshold ($8K-$10K)')
            
            ax.hist(near_ctr['amount'], bins=20, color='#ef4444', alpha=0.8, edgecolor='#991b1b')
            ax.axvline(x=10000, color='#fbbf24', linestyle='--', linewidth=2, label='$10K CTR Limit')
            ax.set_xlabel('Amount ($)')
            ax.set_ylabel('Frequency')
            ax.legend(facecolor='#1a1f35', edgecolor='#374151', labelcolor='white')
            charts.append(_fig_to_base64(fig))
    
    # ── Build Summary ──
    duration_ms = (time.time() - start_time) * 1000
    
    summary = "\n".join(summaries)
    summary += f"\n\n⏱️ EDA completed in {duration_ms:.0f}ms, generated {len(charts)} charts."
    
    return {
        'summary': summary,
        'statistics': stats,
        'charts': charts,
        'duration_ms': duration_ms
    }
