"""
Enumerations for AML Sentinel.
"""

from enum import Enum


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EscalationAction(str, Enum):
    NO_ACTION = "no_action"
    MONITOR = "monitor"
    REVIEW = "flag_for_review"
    REPORT = "file_sar"


class QueryIntent(str, Enum):
    FULL_ANALYSIS = "full_analysis"
    PATTERN_SEARCH = "pattern_search"
    THRESHOLD_QUERY = "threshold_query"
    ENTITY_LOOKUP = "entity_lookup"
    COMPARISON = "comparison"
    EDA_REQUEST = "eda_request"
    AGGREGATION = "aggregation"
    GENERAL = "general"


class AMLPattern(str, Enum):
    STRUCTURING = "structuring"
    SMURFING = "smurfing"
    LAYERING = "layering"
    RAPID_CASHOUT = "rapid_cashout"
    ROUND_TRIP = "round_trip"
    VELOCITY_SPIKE = "velocity_spike"
    DORMANT_ACTIVATION = "dormant_activation"
    GEOGRAPHIC_RISK = "geographic_risk"
    ALL = "all"


class ToolName(str, Enum):
    EDA = "eda_tool"
    FEATURE_ENGINEERING = "feature_engineering"
    ANOMALY_DETECTION = "anomaly_detection"
    RISK_CLASSIFICATION = "risk_classification"
    EXPLANATION = "explanation_engine"
