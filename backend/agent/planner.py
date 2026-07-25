"""
Dynamic Planner — Constructs execution plans based on parsed query intent.

Maps parsed query intents to the minimal set of tools needed,
determining the order of execution and data subsets to operate on.
"""

from models.enums import QueryIntent, AMLPattern, ToolName
from models.schemas import ParsedQuery


def build_execution_plan(parsed_query: ParsedQuery) -> list[str]:
    """
    Build a dynamic execution plan based on the parsed query.
    
    Returns a list of tool names to execute in order.
    The plan is DYNAMIC — different queries produce different plans.
    """
    plan = []
    intent = parsed_query.intent
    
    # ── Full Analysis: Run everything ──
    if intent == QueryIntent.FULL_ANALYSIS:
        plan = [
            ToolName.EDA.value,
            ToolName.FEATURE_ENGINEERING.value,
            ToolName.ANOMALY_DETECTION.value,
            ToolName.RISK_CLASSIFICATION.value,
            ToolName.EXPLANATION.value,
        ]
    
    # ── EDA Request: Only EDA ──
    elif intent == QueryIntent.EDA_REQUEST:
        plan = [ToolName.EDA.value]
    
    # ── Pattern Search: Feature engineering + detection + classification + explain ──
    elif intent == QueryIntent.PATTERN_SEARCH:
        if parsed_query.requires_eda:
            plan.append(ToolName.EDA.value)
        plan.extend([
            ToolName.FEATURE_ENGINEERING.value,
            ToolName.ANOMALY_DETECTION.value,
            ToolName.RISK_CLASSIFICATION.value,
            ToolName.EXPLANATION.value,
        ])
    
    # ── Entity Lookup: Focused analysis on specific customer(s) ──
    elif intent == QueryIntent.ENTITY_LOOKUP:
        # No EDA needed for single-entity lookup
        plan = [
            ToolName.FEATURE_ENGINEERING.value,
            ToolName.ANOMALY_DETECTION.value,
            ToolName.RISK_CLASSIFICATION.value,
            ToolName.EXPLANATION.value,
        ]
    
    # ── Comparison: Side-by-side entity analysis ──
    elif intent == QueryIntent.COMPARISON:
        plan = [
            ToolName.FEATURE_ENGINEERING.value,
            ToolName.ANOMALY_DETECTION.value,
            ToolName.RISK_CLASSIFICATION.value,
            ToolName.EXPLANATION.value,
        ]
    
    # ── Threshold Query: Direct aggregation, minimal ML ──
    elif intent == QueryIntent.THRESHOLD_QUERY:
        # For threshold queries, we might not need full ML pipeline
        if parsed_query.requires_anomaly_detection:
            plan = [
                ToolName.FEATURE_ENGINEERING.value,
                ToolName.ANOMALY_DETECTION.value,
                ToolName.RISK_CLASSIFICATION.value,
                ToolName.EXPLANATION.value,
            ]
        else:
            plan = [ToolName.FEATURE_ENGINEERING.value]
    
    # ── Aggregation: Data summary queries ──
    elif intent == QueryIntent.AGGREGATION:
        plan = [ToolName.EDA.value]
    
    # ── Default: Full pipeline ──
    else:
        plan = [
            ToolName.EDA.value,
            ToolName.FEATURE_ENGINEERING.value,
            ToolName.ANOMALY_DETECTION.value,
            ToolName.RISK_CLASSIFICATION.value,
            ToolName.EXPLANATION.value,
        ]
    
    return plan


def get_plan_description(plan: list[str], parsed_query: ParsedQuery) -> str:
    """Generate a human-readable description of the execution plan."""
    
    tool_descriptions = {
        ToolName.EDA.value: "Exploratory Data Analysis",
        ToolName.FEATURE_ENGINEERING.value: "AML Feature Engineering",
        ToolName.ANOMALY_DETECTION.value: "Anomaly Detection (ML + Rules)",
        ToolName.RISK_CLASSIFICATION.value: "Risk Classification",
        ToolName.EXPLANATION.value: "Explanation Generation",
    }
    
    intent_descriptions = {
        QueryIntent.FULL_ANALYSIS: "Full dataset analysis for suspicious activity",
        QueryIntent.PATTERN_SEARCH: f"Searching for specific pattern(s): {', '.join(str(p) for p in parsed_query.target_patterns)}",
        QueryIntent.ENTITY_LOOKUP: f"Focused analysis on customer(s): {', '.join(parsed_query.target_entities)}",
        QueryIntent.COMPARISON: f"Comparing entities: {', '.join(parsed_query.target_entities)}",
        QueryIntent.THRESHOLD_QUERY: "Threshold-based query",
        QueryIntent.EDA_REQUEST: "Exploratory data analysis",
        QueryIntent.AGGREGATION: "Data aggregation query",
    }
    
    lines = [
        f"📋 Intent: {intent_descriptions.get(parsed_query.intent, str(parsed_query.intent))}",
        f"🔧 Execution Plan ({len(plan)} steps):",
    ]
    
    for i, tool in enumerate(plan, 1):
        desc = tool_descriptions.get(tool, tool)
        lines.append(f"   {i}. {desc}")
    
    # Note what was skipped
    all_tools = set(t.value for t in ToolName)
    skipped = all_tools - set(plan)
    if skipped:
        skipped_names = [tool_descriptions.get(t, t) for t in skipped]
        lines.append(f"⏭️ Skipped: {', '.join(skipped_names)}")
    
    # Note filters
    f = parsed_query.filters
    filter_notes = []
    if f.customer_ids:
        filter_notes.append(f"Customers: {', '.join(f.customer_ids)}")
    if f.date_start or f.date_end:
        filter_notes.append(f"Date: {f.date_start or 'start'} → {f.date_end or 'now'}")
    if f.min_amount is not None or f.max_amount is not None:
        filter_notes.append(f"Amount: {f'${f.min_amount:,.0f}' if f.min_amount else '$0'} – {f'${f.max_amount:,.0f}' if f.max_amount else '∞'}")
    if f.transaction_types:
        filter_notes.append(f"Types: {', '.join(f.transaction_types)}")
    if f.min_transaction_count:
        filter_notes.append(f"Min transactions: {f.min_transaction_count}+")
    
    if filter_notes:
        lines.append(f"🔍 Filters: {' | '.join(filter_notes)}")
    
    return "\n".join(lines)
