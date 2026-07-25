"""
Query Parser — Extracts structured intent from natural language queries.

Uses LLM (Google Gemini) to parse user queries into structured intent,
filters, entities, and pattern types. Falls back to keyword-based
parsing when LLM is unavailable.
"""

import re
import json
from typing import Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.enums import QueryIntent, AMLPattern
from models.schemas import ParsedQuery, QueryFilters


def _keyword_parse(query: str, max_date=None) -> ParsedQuery:
    """
    Fallback keyword-based query parser.
    Used when LLM is unavailable or for fast parsing.
    """
    q = query.lower().strip()
    
    filters = QueryFilters()
    intent = QueryIntent.GENERAL
    target_entities = []
    target_patterns = []
    
    requires_eda = False
    requires_fe = False
    requires_ad = False
    requires_rc = False
    
    # ── Detect Intent ──
    
    # Entity lookup: "Is customer C-4521 suspicious?" or "check customer C-1234"
    customer_pattern = re.findall(r'(?:customer|cust|id)\s*(?:id\s*)?[#:]?\s*(C-?\d+|\d{3,})', q, re.IGNORECASE)
    if not customer_pattern:
        customer_pattern = re.findall(r'(C-\d+)', query)  # Case-sensitive for C-XXXX
    
    if customer_pattern:
        target_entities = [cid.upper() if cid.upper().startswith('C-') else f'C-{cid.upper()}' for cid in customer_pattern]
        filters.customer_ids = target_entities
        
        if len(target_entities) == 1:
            intent = QueryIntent.ENTITY_LOOKUP
            requires_fe = True
            requires_ad = True
            requires_rc = True
        elif 'compare' in q or 'vs' in q or 'versus' in q:
            intent = QueryIntent.COMPARISON
            requires_fe = True
            requires_ad = True
            requires_rc = True
        else:
            intent = QueryIntent.ENTITY_LOOKUP
            requires_fe = True
            requires_ad = True
            requires_rc = True
    
    # Full analysis
    elif any(phrase in q for phrase in ['analyze', 'analyse', 'full analysis', 'scan', 
                                         'suspicious activity', 'detect suspicious',
                                         'find suspicious', 'check for']):
        intent = QueryIntent.FULL_ANALYSIS
        requires_eda = True
        requires_fe = True
        requires_ad = True
        requires_rc = True
    
    # EDA-specific
    elif any(phrase in q for phrase in ['eda', 'exploratory', 'distribution', 'profile',
                                         'overview', 'summary of data', 'data summary',
                                         'describe the data', 'dataset info']):
        intent = QueryIntent.EDA_REQUEST
        requires_eda = True
    
    # Pattern-specific search
    elif any(phrase in q for phrase in ['structuring', 'smurfing', 'smurf']):
        intent = QueryIntent.PATTERN_SEARCH
        target_patterns = [AMLPattern.STRUCTURING]
        requires_fe = True
        requires_ad = True
        requires_rc = True
    
    elif any(phrase in q for phrase in ['layering', 'rapid movement', 'rapid cash', 
                                         'pass-through', 'pass through']):
        intent = QueryIntent.PATTERN_SEARCH
        target_patterns = [AMLPattern.RAPID_CASHOUT]
        requires_fe = True
        requires_ad = True
        requires_rc = True
    
    elif any(phrase in q for phrase in ['velocity', 'spike', 'surge', 'sudden increase']):
        intent = QueryIntent.PATTERN_SEARCH
        target_patterns = [AMLPattern.VELOCITY_SPIKE]
        requires_fe = True
        requires_ad = True
        requires_rc = True
    
    elif any(phrase in q for phrase in ['dormant', 'inactive', 'reactivat']):
        intent = QueryIntent.PATTERN_SEARCH
        target_patterns = [AMLPattern.DORMANT_ACTIVATION]
        requires_fe = True
        requires_ad = True
        requires_rc = True
    
    elif any(phrase in q for phrase in ['geographic', 'country risk', 'high-risk countr',
                                         'jurisdiction', 'offshore']):
        intent = QueryIntent.PATTERN_SEARCH
        target_patterns = [AMLPattern.GEOGRAPHIC_RISK]
        requires_fe = True
        requires_ad = True
        requires_rc = True
    
    # Threshold / aggregation queries
    elif re.search(r'\d+\+?\s*transaction', q) or 'how many' in q or 'count' in q:
        intent = QueryIntent.THRESHOLD_QUERY
        
        # Extract threshold number
        num_match = re.search(r'(\d+)\+?\s*(?:transaction|deposit|transfer|withdrawal)', q)
        if num_match:
            filters.min_transaction_count = int(num_match.group(1))
    
    elif any(phrase in q for phrase in ['average', 'total', 'sum', 'max', 'min', 'top']):
        intent = QueryIntent.AGGREGATION
    
    # Wire transfer focus
    elif 'wire' in q:
        intent = QueryIntent.PATTERN_SEARCH
        filters.transaction_types = ['wire_transfer']
        requires_fe = True
        requires_ad = True
        requires_rc = True
    
    # High risk focus
    elif 'high risk' in q or 'high-risk' in q or 'risky' in q:
        intent = QueryIntent.PATTERN_SEARCH
        target_patterns = [AMLPattern.ALL]
        requires_fe = True
        requires_ad = True
        requires_rc = True
    
    # ── Extract Filters ──
    
    # Date range
    date_patterns = [
        (r'last\s+(\d+)\s+days?', 'days'),
        (r'past\s+(\d+)\s+days?', 'days'),
        (r'last\s+(\d+)\s+months?', 'months'),
        (r'last\s+(\d+)\s+weeks?', 'weeks'),
    ]
    for pattern, unit in date_patterns:
        match = re.search(pattern, q)
        if match:
            num = int(match.group(1))
            from datetime import datetime, timedelta
            end = max_date if max_date else datetime.now()
            if unit == 'days':
                start = end - timedelta(days=num)
            elif unit == 'weeks':
                start = end - timedelta(weeks=num)
            elif unit == 'months':
                start = end - timedelta(days=num * 30)
            filters.date_start = start.strftime('%Y-%m-%d')
            filters.date_end = end.strftime('%Y-%m-%d')
            break
    
    # Amount filters
    amount_match = re.search(r'(?:under|below|less than)\s*\$?([\d,]+)', q)
    if amount_match:
        filters.max_amount = float(amount_match.group(1).replace(',', ''))
    
    amount_match = re.search(r'(?:over|above|more than|exceeding)\s*\$?([\d,]+)', q)
    if amount_match:
        filters.min_amount = float(amount_match.group(1).replace(',', ''))
    
    # Transaction type filters
    if 'cash' in q and 'deposit' in q:
        filters.transaction_types = ['deposit']
    elif 'wire' in q and 'transfer' in q:
        filters.transaction_types = ['wire_transfer']
    elif 'transfer' in q:
        filters.transaction_types = ['transfer', 'wire_transfer']
    
    # Default: if nothing specific detected, do full analysis
    if intent == QueryIntent.GENERAL:
        intent = QueryIntent.FULL_ANALYSIS
        requires_eda = True
        requires_fe = True
        requires_ad = True
        requires_rc = True
    
    return ParsedQuery(
        original_query=query,
        intent=intent,
        filters=filters,
        target_entities=target_entities,
        target_patterns=target_patterns,
        requires_eda=requires_eda,
        requires_feature_engineering=requires_fe,
        requires_anomaly_detection=requires_ad,
        requires_risk_classification=requires_rc,
        requires_explanation=True,
    )


async def parse_query_with_llm(query: str, llm=None, max_date=None) -> ParsedQuery:
    """
    Parse query using LLM for better understanding.
    Falls back to keyword parsing if LLM fails.
    """
    if llm is None:
        return _keyword_parse(query, max_date)
    
    try:
        anchor_info = f"\nNote: The current 'today' date anchor is {max_date.strftime('%Y-%m-%d')} for resolving relative dates." if max_date else ""
        prompt = f"""You are an AML (Anti-Money Laundering) query parser. Analyze the following user query and extract structured information.{anchor_info}

User Query: "{query}"

Return a JSON object with:
{{
    "intent": one of ["full_analysis", "pattern_search", "threshold_query", "entity_lookup", "comparison", "eda_request", "aggregation"],
    "customer_ids": list of customer IDs mentioned (format: "C-XXXX"),
    "patterns": list of AML patterns to look for: ["structuring", "smurfing", "layering", "rapid_cashout", "velocity_spike", "dormant_activation", "geographic_risk", "all"],
    "date_start": start date if mentioned (YYYY-MM-DD format or null),
    "date_end": end date if mentioned (YYYY-MM-DD format or null),
    "min_amount": minimum amount filter or null,
    "max_amount": maximum amount filter or null,
    "transaction_types": list of types mentioned: ["deposit", "withdrawal", "transfer", "wire_transfer"],
    "min_transaction_count": minimum transaction count threshold or null,
    "requires_eda": boolean,
    "requires_feature_engineering": boolean,
    "requires_anomaly_detection": boolean,
    "requires_risk_classification": boolean
}}

Return ONLY the JSON, no markdown.
"""
        
        response = await llm.ainvoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Clean up response
        content = content.strip()
        if content.startswith('```'):
            content = content.split('\n', 1)[1]
            if content.endswith('```'):
                content = content[:-3]
        
        data = json.loads(content)
        
        filters = QueryFilters(
            customer_ids=data.get('customer_ids', []),
            date_start=data.get('date_start'),
            date_end=data.get('date_end'),
            min_amount=data.get('min_amount'),
            max_amount=data.get('max_amount'),
            transaction_types=data.get('transaction_types', []),
            min_transaction_count=data.get('min_transaction_count'),
        )
        
        patterns = []
        for p in data.get('patterns', []):
            try:
                patterns.append(AMLPattern(p))
            except ValueError:
                pass
        
        return ParsedQuery(
            original_query=query,
            intent=QueryIntent(data.get('intent', 'full_analysis')),
            filters=filters,
            target_entities=data.get('customer_ids', []),
            target_patterns=patterns,
            requires_eda=data.get('requires_eda', False),
            requires_feature_engineering=data.get('requires_feature_engineering', True),
            requires_anomaly_detection=data.get('requires_anomaly_detection', True),
            requires_risk_classification=data.get('requires_risk_classification', True),
            requires_explanation=True,
        )
    
    except Exception as e:
        print(f"LLM parsing failed ({e}), falling back to keyword parser")
        return _keyword_parse(query, max_date)


def parse_query(query: str, max_date=None) -> ParsedQuery:
    """Wrapper for keyword parser."""
    return _keyword_parse(query, max_date)
