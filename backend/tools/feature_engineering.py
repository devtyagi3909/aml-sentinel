"""
feature_engineering.py

Constructs higher-order, derived representations of raw financial transactions suitable for 
machine learning anomaly detection and heuristic rule evaluation. Extrapolates temporal 
velocity, spatial geo-risk, structuring indicators, and graph-theoretical centrality metrics.
"""

import pandas as pd
import numpy as np
import time
from typing import Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    CTR_THRESHOLD, STRUCTURING_LOWER, STRUCTURING_UPPER,
    STRUCTURING_WINDOW_DAYS, VELOCITY_SPIKE_MULTIPLIER,
    RAPID_MOVEMENT_HOURS, DORMANCY_DAYS, HIGH_RISK_COUNTRIES
)


def engineer_features(
    transactions: pd.DataFrame,
    customers: Optional[pd.DataFrame] = None,
    customer_ids: Optional[list] = None,
    feature_set: str = "all",
    pattern_focus: Optional[str] = None,
) -> dict:
    """
    Generate AML-specific features per customer from transaction data.
    
    Args:
        transactions: Transaction DataFrame
        customers: Customer DataFrame (optional, for enrichment)
        customer_ids: Compute features only for these customers
        feature_set: "all", "structuring", "velocity", "geographic", "behavioral"
        pattern_focus: Focus on specific AML pattern features
        
    Returns:
        dict with 'features_df' (pd.DataFrame), 'summary' (str), 'feature_names' (list)
    """
    start_time = time.time()
    df = transactions.copy()
    
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns and df['timestamp'].dtype == 'object':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter to specific customers if requested
    if customer_ids:
        df = df[df['customer_id'].isin(customer_ids)]
    
    if len(df) == 0:
        return {
            'features_df': pd.DataFrame(),
            'summary': 'No transactions found for the specified customers.',
            'feature_names': [],
            'duration_ms': (time.time() - start_time) * 1000
        }
    
    # Sort by customer and time
    df = df.sort_values(['customer_id', 'timestamp'])
    
    features_list = []
    feature_names_used = []
    
    # ── Customer-Level Feature Aggregation ──
    customer_groups = df.groupby('customer_id')
    
    # === CORE FEATURES (always computed) ===
    core_features = customer_groups.agg(
        total_transactions=('transaction_id', 'count'),
        total_amount=('amount', 'sum'),
        avg_amount=('amount', 'mean'),
        std_amount=('amount', 'std'),
        max_amount=('amount', 'max'),
        min_amount=('amount', 'min'),
        median_amount=('amount', 'median'),
    ).fillna(0)
    
    feature_names_used.extend(core_features.columns.tolist())
    
    # === STRUCTURING FEATURES ===
    if feature_set in ("all", "structuring") or pattern_focus == "structuring":
        cash_df = df[df.get('is_cash', pd.Series(False, index=df.index)) == True] if 'is_cash' in df.columns else pd.DataFrame()
        
        if len(cash_df) > 0:
            # Count of cash transactions near CTR threshold
            near_ctr = cash_df[
                (cash_df['amount'] >= STRUCTURING_LOWER) &
                (cash_df['amount'] < CTR_THRESHOLD)
            ]
            near_ctr_counts = near_ctr.groupby('customer_id').size().rename('cash_near_ctr_count')
            
            # Total cash amount near CTR
            near_ctr_total = near_ctr.groupby('customer_id')['amount'].sum().rename('cash_near_ctr_total')
            
            # Max clustering of near-CTR transactions within STRUCTURING_WINDOW_DAYS
            def max_cluster_count(group):
                if len(group) < 2:
                    return len(group)
                dates = group['timestamp'].sort_values()
                max_count = 0
                for i in range(len(dates)):
                    window_end = dates.iloc[i] + pd.Timedelta(days=STRUCTURING_WINDOW_DAYS)
                    count = ((dates >= dates.iloc[i]) & (dates <= window_end)).sum()
                    max_count = max(max_count, count)
                return max_count
            
            # Average gap between near-CTR transactions
            def avg_gap(group):
                if len(group) < 2:
                    return 999.0
                dates = group['timestamp'].sort_values()
                gaps = dates.diff().dt.total_seconds() / 3600  # hours
                return gaps.mean()

            if near_ctr.empty:
                structuring_cluster = pd.Series(name='structuring_cluster_max', dtype=float)
                near_ctr_gap = pd.Series(name='near_ctr_avg_gap_hours', dtype=float)
            else:
                s_clust = near_ctr.groupby('customer_id').apply(max_cluster_count, include_groups=False)
                structuring_cluster = s_clust if isinstance(s_clust, pd.DataFrame) else s_clust.rename('structuring_cluster_max')
                if isinstance(structuring_cluster, pd.DataFrame):
                    structuring_cluster = pd.Series(name='structuring_cluster_max', dtype=float)
                    
                n_gap = near_ctr.groupby('customer_id').apply(avg_gap, include_groups=False)
                near_ctr_gap = n_gap if isinstance(n_gap, pd.DataFrame) else n_gap.rename('near_ctr_avg_gap_hours')
                if isinstance(near_ctr_gap, pd.DataFrame):
                    near_ctr_gap = pd.Series(name='near_ctr_avg_gap_hours', dtype=float)
            
            # Total cash transactions
            total_cash = cash_df.groupby('customer_id').size().rename('total_cash_transactions')
            cash_pct = (total_cash / core_features['total_transactions']).rename('cash_transaction_pct').fillna(0)
            
            for feat in [near_ctr_counts, near_ctr_total, structuring_cluster, near_ctr_gap, total_cash, cash_pct]:
                core_features = core_features.join(feat, how='left')
                feature_names_used.append(feat.name)
        else:
            for name in ['cash_near_ctr_count', 'cash_near_ctr_total', 'structuring_cluster_max',
                         'near_ctr_avg_gap_hours', 'total_cash_transactions', 'cash_transaction_pct']:
                core_features[name] = 0
                feature_names_used.append(name)
    
    # === VELOCITY FEATURES ===
    if feature_set in ("all", "velocity") or pattern_focus == "velocity_spike":
        # Transaction frequency in different windows
        now = df['timestamp'].max()
        
        for days, label in [(1, '1d'), (7, '7d'), (30, '30d')]:
            window_start = now - pd.Timedelta(days=days)
            window_df = df[df['timestamp'] >= window_start]
            window_counts = window_df.groupby('customer_id').size().rename(f'txn_count_{label}')
            window_amounts = window_df.groupby('customer_id')['amount'].sum().rename(f'txn_volume_{label}')
            core_features = core_features.join(window_counts, how='left').fillna(0)
            core_features = core_features.join(window_amounts, how='left').fillna(0)
            feature_names_used.extend([f'txn_count_{label}', f'txn_volume_{label}'])
        
        # Velocity ratio: recent activity vs historical average
        total_days = max((now - df['timestamp'].min()).days, 1)
        daily_avg = core_features['total_transactions'] / total_days
        core_features['velocity_ratio_7d'] = (core_features.get('txn_count_7d', 0) / 7) / daily_avg.replace(0, 0.001)
        core_features['velocity_ratio_30d'] = (core_features.get('txn_count_30d', 0) / 30) / daily_avg.replace(0, 0.001)
        feature_names_used.extend(['velocity_ratio_7d', 'velocity_ratio_30d'])
        
        # Time between transactions (mean and min)
        def txn_timing_stats(group):
            if len(group) < 2:
                return pd.Series({'mean_gap_hours': 999.0, 'min_gap_hours': 999.0})
            gaps = group['timestamp'].sort_values().diff().dropna().dt.total_seconds() / 3600
            return pd.Series({
                'mean_gap_hours': gaps.mean(),
                'min_gap_hours': gaps.min()
            })
        
        timing = customer_groups.apply(txn_timing_stats, include_groups=False)
        if isinstance(timing, pd.DataFrame):
            core_features = core_features.join(timing, how='left')
        feature_names_used.extend(['mean_gap_hours', 'min_gap_hours'])
    
    # === GEOGRAPHIC RISK FEATURES ===
    if feature_set in ("all", "geographic") or pattern_focus == "geographic_risk":
        if 'counterparty_country' in df.columns:
            # Count of transactions to high-risk countries
            high_risk_txns = df[df['counterparty_country'].isin(HIGH_RISK_COUNTRIES)]
            hr_counts = high_risk_txns.groupby('customer_id').size().rename('high_risk_country_txns')
            hr_amounts = high_risk_txns.groupby('customer_id')['amount'].sum().rename('high_risk_country_volume')
            
            core_features = core_features.join(hr_counts, how='left').fillna(0)
            core_features = core_features.join(hr_amounts, how='left').fillna(0)
            
            core_features['high_risk_country_pct'] = (
                core_features['high_risk_country_txns'] / core_features['total_transactions']
            ).fillna(0)
            
            # Unique counterparty countries
            unique_countries = df.groupby('customer_id')['counterparty_country'].nunique().rename('unique_countries')
            core_features = core_features.join(unique_countries, how='left').fillna(0)
            
            feature_names_used.extend(['high_risk_country_txns', 'high_risk_country_volume',
                                       'high_risk_country_pct', 'unique_countries'])
    
    # === BEHAVIORAL FEATURES ===
    if feature_set in ("all", "behavioral") or pattern_focus in ("rapid_cashout", "dormant_activation", "layering"):
        # Rapid in-out: large deposit followed by transfer within 24h
        def detect_rapid_movement(group):
            if len(group) < 2:
                return 0
            group = group.sort_values('timestamp')
            count = 0
            deposits = group[group['transaction_type'].isin(['deposit'])]
            transfers = group[group['transaction_type'].isin(['transfer', 'wire_transfer'])]
            
            for _, dep in deposits.iterrows():
                if dep['amount'] < 20000:
                    continue
                window_end = dep['timestamp'] + pd.Timedelta(hours=RAPID_MOVEMENT_HOURS)
                rapid_outs = transfers[
                    (transfers['timestamp'] > dep['timestamp']) &
                    (transfers['timestamp'] <= window_end) &
                    (transfers['amount'] >= dep['amount'] * 0.7)
                ]
                count += len(rapid_outs)
            return count
        
        rapid_movement = customer_groups.apply(
            detect_rapid_movement, include_groups=False
        ).rename('rapid_movement_count')
        core_features = core_features.join(rapid_movement, how='left').fillna(0)
        feature_names_used.append('rapid_movement_count')
        
        # Dormancy detection
        def dormancy_indicator(group):
            if len(group) < 2:
                return 0
            dates = group['timestamp'].sort_values()
            gaps = dates.diff().dt.days.dropna()
            max_gap = gaps.max()
            return 1 if max_gap >= DORMANCY_DAYS else 0
        
        dormancy = customer_groups.apply(
            dormancy_indicator, include_groups=False
        ).rename('dormancy_flag')
        core_features = core_features.join(dormancy, how='left').fillna(0)
        feature_names_used.append('dormancy_flag')
        
        # Weekend/night transaction percentage
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5
            df['is_unusual_time'] = (df['hour'] < 6) | (df['hour'] > 22) | df['is_weekend']
            
            unusual_pct = df.groupby('customer_id')['is_unusual_time'].mean().rename('unusual_time_pct')
            core_features = core_features.join(unusual_pct, how='left').fillna(0)
            feature_names_used.append('unusual_time_pct')
        
        # Amount deviation (z-score of customer's recent transactions vs their own history)
        core_features['amount_cv'] = (core_features['std_amount'] / core_features['avg_amount'].replace(0, 1)).fillna(0)
        feature_names_used.append('amount_cv')
    
    # === GRAPH FEATURES (NETWORKX) ===
    import networkx as nx
    
    # Build a directed graph of transactions to detect complex laundering networks
    transfer_df = df[df['transaction_type'].isin(['transfer', 'wire_transfer']) & (df['counterparty_id'] != '') & (df['counterparty_id'].notna())]
    if len(transfer_df) > 0:
        G = nx.from_pandas_edgelist(
            transfer_df, 
            source='customer_id', 
            target='counterparty_id', 
            create_using=nx.DiGraph()
        )
        
        # Calculate PageRank (identifies central hubs / money mules)
        pagerank = nx.pagerank(G, alpha=0.85)
        pr_series = pd.Series(pagerank, name='network_pagerank')
        core_features = core_features.join(pr_series, how='left')
        feature_names_used.append('network_pagerank')
        
        # Calculate degree metrics
        in_degree = pd.Series(dict(G.in_degree()), name='network_in_degree')
        out_degree = pd.Series(dict(G.out_degree()), name='network_out_degree')
        core_features = core_features.join(in_degree, how='left')
        core_features = core_features.join(out_degree, how='left')
        feature_names_used.extend(['network_in_degree', 'network_out_degree'])
        
        # Detect circular flows (Round Tripping) via simple cycles
        # Note: simple_cycles is O((V+E) * C) which grows exponentially. 
        # For real-time UX, we only run it if the graph is reasonably small.
        try:
            if G.number_of_edges() < 2000:
                cycles = list(nx.simple_cycles(G, length_bound=4)) 
                cycle_counts = {}
                for cycle in cycles:
                    if len(cycle) >= 2:
                        for node in cycle:
                            cycle_counts[node] = cycle_counts.get(node, 0) + 1
                
                cycle_series = pd.Series(cycle_counts, name='circular_flow_count')
                core_features = core_features.join(cycle_series, how='left')
                feature_names_used.append('circular_flow_count')
        except Exception:
            pass
            
        core_features['network_pagerank'] = core_features['network_pagerank'].fillna(0)
        core_features['network_in_degree'] = core_features['network_in_degree'].fillna(0)
        core_features['network_out_degree'] = core_features['network_out_degree'].fillna(0)
        if 'circular_flow_count' in core_features.columns:
            core_features['circular_flow_count'] = core_features['circular_flow_count'].fillna(0)
    else:
        for name in ['network_pagerank', 'network_in_degree', 'network_out_degree', 'circular_flow_count']:
            core_features[name] = 0.0
            feature_names_used.append(name)
            
    # ── Enrich with customer data if available ──
    if customers is not None and len(customers) > 0:
        cust = customers.set_index('customer_id')
        for col in ['risk_category', 'pep_flag', 'kyc_status', 'account_type']:
            if col in cust.columns:
                core_features = core_features.join(cust[[col]], how='left')
                feature_names_used.append(col)
        
        # Encode categoricals for ML
        if 'risk_category' in core_features.columns:
            risk_map = {'low': 0, 'medium': 1, 'high': 2}
            core_features['risk_category_encoded'] = core_features['risk_category'].map(risk_map).fillna(0)
            feature_names_used.append('risk_category_encoded')
        
        if 'pep_flag' in core_features.columns:
            core_features['pep_flag'] = core_features['pep_flag'].fillna(0).astype(int)
        
        if 'kyc_status' in core_features.columns:
            kyc_map = {'verified': 0, 'pending': 1, 'expired': 2}
            core_features['kyc_status_encoded'] = core_features['kyc_status'].map(kyc_map).fillna(0)
            feature_names_used.append('kyc_status_encoded')
    
    # Fill NaN
    core_features = core_features.fillna(0)
    
    # Deduplicate feature names
    feature_names_used = list(dict.fromkeys(feature_names_used))
    
    duration_ms = (time.time() - start_time) * 1000
    
    summary_lines = [
        f"Feature engineering completed in {duration_ms:.0f}ms",
        f"Generated {len(feature_names_used)} features for {len(core_features)} customers",
        f"Feature set: {feature_set}" + (f" (focused on {pattern_focus})" if pattern_focus else ""),
    ]
    
    # Highlight interesting findings
    if 'cash_near_ctr_count' in core_features.columns:
        structuring_suspects = (core_features['cash_near_ctr_count'] >= 3).sum()
        if structuring_suspects > 0:
            summary_lines.append(f"ALERT: {structuring_suspects} customers have 3+ near-CTR cash transactions")
    
    if 'rapid_movement_count' in core_features.columns:
        rapid_suspects = (core_features['rapid_movement_count'] > 0).sum()
        if rapid_suspects > 0:
            summary_lines.append(f"ALERT: {rapid_suspects} customers show rapid deposit-then-transfer patterns")
    
    if 'high_risk_country_pct' in core_features.columns:
        geo_suspects = (core_features['high_risk_country_pct'] > 0.3).sum()
        if geo_suspects > 0:
            summary_lines.append(f"ALERT: {geo_suspects} customers have >30% transactions to high-risk countries")
    
    return {
        'features_df': core_features,
        'summary': "\n".join(summary_lines),
        'feature_names': feature_names_used,
        'duration_ms': duration_ms
    }
