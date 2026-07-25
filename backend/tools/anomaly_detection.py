"""
Anomaly Detection Tool — Identifies suspicious transactions and customers.

Uses a hybrid approach combining:
1. Isolation Forest (ML-based unsupervised anomaly detection)
2. Statistical Z-Score analysis
3. Rule-based AML pattern matching
4. Ensemble scoring combining all methods
"""

import pandas as pd
import numpy as np
import time
from typing import Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    CTR_THRESHOLD, STRUCTURING_LOWER, STRUCTURING_UPPER,
    STRUCTURING_COUNT_THRESHOLD, VELOCITY_SPIKE_MULTIPLIER,
    HIGH_RISK_COUNTRIES
)


def detect_anomalies(
    features_df: pd.DataFrame,
    transactions: Optional[pd.DataFrame] = None,
    method: str = "ensemble",
    pattern_focus: Optional[str] = None,
    sensitivity: float = 0.1,
    customer_ids: Optional[list] = None,
) -> dict:
    """
    Detect anomalous patterns in customer features.
    
    Args:
        features_df: Customer feature DataFrame (from feature engineering)
        transactions: Original transactions (for rule-based checks)
        method: "isolation_forest", "zscore", "rules", "ensemble"
        pattern_focus: Focus on specific pattern (e.g., "structuring")
        sensitivity: Contamination factor for Isolation Forest (0.01-0.5)
        customer_ids: Analyze only these customers
        
    Returns:
        dict with 'scores_df', 'summary', 'anomaly_count', 'method_details'
    """
    start_time = time.time()
    
    df = features_df.copy()
    
    if customer_ids:
        df = df[df.index.isin(customer_ids)] if df.index.name == 'customer_id' else df[df['customer_id'].isin(customer_ids)]
    
    if len(df) == 0:
        return {
            'scores_df': pd.DataFrame(),
            'summary': 'No data available for anomaly detection.',
            'anomaly_count': 0,
            'method_details': {},
            'duration_ms': (time.time() - start_time) * 1000
        }
    
    # Initialize scores DataFrame
    scores = pd.DataFrame(index=df.index)
    scores.index.name = 'customer_id'
    method_details = {}
    
    # Select numeric features for ML
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove encoded/flag columns that are already rule-based
    ml_features = [c for c in numeric_cols if c not in [
        'risk_category_encoded', 'kyc_status_encoded', 'pep_flag',
        'dormancy_flag', 'rapid_movement_count'
    ]]
    
    if len(ml_features) == 0:
        ml_features = numeric_cols[:5]  # fallback
    
    # ── 1. ISOLATION FOREST ──
    if method in ("isolation_forest", "ensemble"):
        try:
            X = df[ml_features].fillna(0).values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            contamination = min(max(sensitivity, 0.01), 0.3)
            
            iso_forest = IsolationForest(
                n_estimators=200,
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
            
            iso_forest.fit(X_scaled)
            
            # Raw anomaly scores (more negative = more anomalous)
            raw_scores = iso_forest.decision_function(X_scaled)
            # Normalize to 0-1 (1 = most anomalous)
            iso_scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-10)
            
            scores['isolation_forest_score'] = iso_scores
            
            iso_anomalies = (iso_forest.predict(X_scaled) == -1).sum()
            method_details['isolation_forest'] = {
                'anomalies_detected': int(iso_anomalies),
                'contamination': contamination,
                'features_used': len(ml_features),
                'description': f'Isolation Forest detected {iso_anomalies} anomalies using {len(ml_features)} features'
            }
        except Exception as e:
            scores['isolation_forest_score'] = 0.0
            method_details['isolation_forest'] = {'error': str(e)}
    
    # ── 2. STATISTICAL Z-SCORE ──
    if method in ("zscore", "ensemble"):
        try:
            X = df[ml_features].fillna(0)
            
            # Z-scores for each feature
            means = X.mean()
            stds = X.std().replace(0, 1)
            z_scores = ((X - means) / stds).abs()
            
            # Aggregate: max z-score across features
            max_z = z_scores.max(axis=1)
            # Normalize to 0-1
            z_normalized = max_z.clip(upper=10) / 10  # cap at z=10
            
            scores['zscore_score'] = z_normalized.values
            
            z_anomalies = (max_z > 3).sum()
            method_details['zscore'] = {
                'anomalies_above_3sigma': int(z_anomalies),
                'mean_max_zscore': float(max_z.mean()),
                'description': f'{z_anomalies} customers exceed 3σ threshold on at least one feature'
            }
        except Exception as e:
            scores['zscore_score'] = 0.0
            method_details['zscore'] = {'error': str(e)}
    
    # ── 3. RULE-BASED PATTERN DETECTION ──
    if method in ("rules", "ensemble"):
        rule_scores = pd.Series(0.0, index=df.index)
        rule_flags = {}
        
        # Rule 1: Structuring detection
        if 'cash_near_ctr_count' in df.columns:
            structuring_mask = df['cash_near_ctr_count'] >= STRUCTURING_COUNT_THRESHOLD
            if 'structuring_cluster_max' in df.columns:
                structuring_mask = structuring_mask | (df['structuring_cluster_max'] >= STRUCTURING_COUNT_THRESHOLD)
            
            structuring_score = pd.Series(0.0, index=df.index)
            structuring_score[structuring_mask] = np.clip(
                df.loc[structuring_mask, 'cash_near_ctr_count'] / 10, 0.3, 1.0
            )
            rule_scores += structuring_score * 0.35
            rule_flags['structuring'] = int(structuring_mask.sum())
        
        # Rule 2: Velocity spike
        if 'velocity_ratio_7d' in df.columns:
            velocity_mask = df['velocity_ratio_7d'] >= VELOCITY_SPIKE_MULTIPLIER
            velocity_score = pd.Series(0.0, index=df.index)
            velocity_score[velocity_mask] = np.clip(
                df.loc[velocity_mask, 'velocity_ratio_7d'] / 20, 0.3, 1.0
            )
            rule_scores += velocity_score * 0.2
            rule_flags['velocity_spike'] = int(velocity_mask.sum())
        
        # Rule 3: Rapid movement
        if 'rapid_movement_count' in df.columns:
            rapid_mask = df['rapid_movement_count'] > 0
            rapid_score = pd.Series(0.0, index=df.index)
            rapid_score[rapid_mask] = np.clip(
                df.loc[rapid_mask, 'rapid_movement_count'] / 5, 0.4, 1.0
            )
            rule_scores += rapid_score * 0.25
            rule_flags['rapid_movement'] = int(rapid_mask.sum())
        
        # Rule 4: Geographic risk
        if 'high_risk_country_pct' in df.columns:
            geo_mask = df['high_risk_country_pct'] > 0.2
            geo_score = pd.Series(0.0, index=df.index)
            geo_score[geo_mask] = df.loc[geo_mask, 'high_risk_country_pct'].clip(upper=1.0)
            rule_scores += geo_score * 0.15
            rule_flags['geographic_risk'] = int(geo_mask.sum())
        
        # Rule 5: Dormancy activation
        if 'dormancy_flag' in df.columns:
            dormancy_mask = df['dormancy_flag'] == 1
            rule_scores[dormancy_mask] += 0.2
            rule_flags['dormancy_activation'] = int(dormancy_mask.sum())
        
        # Rule 6: PEP flag amplifier
        if 'pep_flag' in df.columns:
            pep_mask = df['pep_flag'] == 1
            rule_scores[pep_mask] *= 1.3  # Amplify risk for PEPs
            rule_flags['pep_amplified'] = int(pep_mask.sum())
        
        # Rule 7: KYC gaps
        if 'kyc_status_encoded' in df.columns:
            kyc_risk_mask = df['kyc_status_encoded'] >= 1
            rule_scores[kyc_risk_mask] += 0.05
        
        # Normalize rule scores to 0-1
        rule_scores = rule_scores.clip(0, 1)
        scores['rule_score'] = rule_scores.values
        
        rule_anomalies = (rule_scores > 0.3).sum()
        method_details['rules'] = {
            'anomalies_detected': int(rule_anomalies),
            'pattern_counts': rule_flags,
            'description': f'Rule-based detection flagged {rule_anomalies} customers across {len(rule_flags)} pattern types'
        }
    
    # ── 4. ENSEMBLE SCORING ──
    if method == "ensemble":
        weights = {
            'isolation_forest_score': 0.30,
            'zscore_score': 0.25,
            'rule_score': 0.45,  # Rules weighted highest for AML
        }
        
        ensemble_score = pd.Series(0.0, index=scores.index)
        total_weight = 0
        
        for col, weight in weights.items():
            if col in scores.columns:
                ensemble_score += scores[col] * weight
                total_weight += weight
        
        if total_weight > 0:
            ensemble_score /= total_weight
        
        scores['ensemble_score'] = ensemble_score.values
        scores['anomaly_score'] = ensemble_score.values  # Primary score
        
        method_details['ensemble'] = {
            'weights': weights,
            'description': f'Ensemble combining {len(weights)} methods with rule-based emphasis'
        }
    else:
        # Use the single method's score as the primary
        score_col = {
            'isolation_forest': 'isolation_forest_score',
            'zscore': 'zscore_score',
            'rules': 'rule_score'
        }.get(method, 'rule_score')
        scores['anomaly_score'] = scores.get(score_col, 0.0)
    
    # Fill any NaN scores (e.g. from 1-row standard deviations) with 0.0
    scores = scores.fillna(0.0)
    
    # Sort by anomaly score descending
    scores = scores.sort_values('anomaly_score', ascending=False)
    
    # Count anomalies at different thresholds
    anomaly_count = int((scores['anomaly_score'] > 0.3).sum())
    high_risk_count = int((scores['anomaly_score'] > 0.6).sum())
    critical_count = int((scores['anomaly_score'] > 0.8).sum())
    
    duration_ms = (time.time() - start_time) * 1000
    
    summary_lines = [
        f"Anomaly detection completed in {duration_ms:.0f}ms using {method} method",
        f"Analyzed {len(scores)} customers",
        f"Total flagged: {anomaly_count} (score > 0.3)",
        f"High risk: {high_risk_count} | Critical: {critical_count}",
    ]
    
    if 'rules' in method_details and 'pattern_counts' in method_details['rules']:
        patterns = method_details['rules']['pattern_counts']
        pattern_str = ", ".join([f"{k}: {v}" for k, v in patterns.items() if v > 0])
        if pattern_str:
            summary_lines.append(f"📋 Patterns detected: {pattern_str}")
    
    return {
        'scores_df': scores,
        'summary': "\n".join(summary_lines),
        'anomaly_count': anomaly_count,
        'high_risk_count': high_risk_count,
        'critical_count': critical_count,
        'method_details': method_details,
        'duration_ms': duration_ms
    }
