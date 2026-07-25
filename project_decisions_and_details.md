# AML Sentinel: Comprehensive Project Documentation & Decision Log

## 1. Executive Summary
AML Sentinel is an autonomous, agentic Anti-Money Laundering (AML) detection platform. It translates natural language queries into executable analysis pipelines, utilizing a hybrid ML/Rule-based engine to detect complex financial crime topologies. This document catalogs every minute technical detail and architectural decision made during its development.

## 2. Core Architecture Stack & Philosophy
**Decision:** Stick to a strictly "Zero-Dependency Frontend + FastAPI Backend" stack. 
- **Backend:** Python + FastAPI. Chosen for extreme speed, async capabilities, and native Pydantic validation. 
- **Frontend:** Pure HTML5, CSS3, and ES6 Vanilla JavaScript. 
- **Why?** We specifically avoided React, Streamlit, or heavy frameworks to ensure the project remains highly portable, instantly runnable by hackathon judges without `npm install` steps, and blazingly fast.
- **Visuals:** Matplotlib/Seaborn charts are generated on the backend and sent as Base64 PNGs. This offloads computation from the browser and prevents the need for bloated frontend charting libraries like Chart.js or D3. 

## 3. Data Engineering & The "Stratified Sampling" Strategy
**Decision:** Discard naive row-capping in favor of deliberate, stratified sub-sampling.
- **The Problem:** True laundering cases represent <0.1% of transactions. If we simply load the first 50,000 rows of a massive 5M+ row Kaggle dataset, the demo would likely contain zero flagged entities, making the platform look broken.
- **The Solution:** We implemented `stratified_sampler.py`. This script scans the raw dataset, explicitly isolates *all* rows belonging to known laundering topologies, and then blends them with randomly sampled "normal" transactions to reach exactly 50,000 rows. 
- **Result:** The system guarantees that all 6 topologies are present for judges to find, while keeping the total dataset small enough (50k rows) to prevent Out-Of-Memory (OOM) errors and ensure sub-second response times. The dataset is cached to disk as `demo_transactions.csv` to avoid resampling on every run.

## 4. Agentic Orchestration & Intent-Driven Routing
**Decision:** Replace static pipelines with an adaptive, intent-driven orchestrator.
- **The Engine:** The `AMLSentinelAgent` parses user queries (via LLM or Regex fallback) into discrete `QueryIntent` enumerations (e.g., `FULL_ANALYSIS`, `AGGREGATION`, `ENTITY_LOOKUP`).
- **Dynamic Routing:** Instead of running every tool every time, the Orchestrator evaluates the intent. If a user asks "Which customers made 10+ transactions?", the agent categorizes this as `AGGREGATION` and intelligently *skips* Anomaly Detection, Risk Classification, and Explanation generation, favoring a direct Pandas aggregation instead.
- **Execution Trace Logging:** Every tool skipped is logged with `ran: false` and a specific, human-readable reason (e.g., "Skipped ML detection for basic aggregation query"). This proves to the user that the agent is actively "thinking" and adapting its pipeline.

## 5. The Hybrid ML Detection Engine & The 6 Topologies
**Decision:** Use a multi-layered Ensemble approach (Isolation Forest + Z-Scores + Hard Rules) rather than relying on a single black-box model.
- **Why?** In enterprise AML, black-box models are rejected by regulators. Our ensemble allows rules to act as hard triggers while ML catches unknown outliers.

**The 6 Engineered Topologies (Verified):**
1. **Structuring/Smurfing:** Detected by tracking 3-7 cash deposits just under the $10k CTR threshold (e.g., $8,000 - $9,999) within a short window.
2. **Velocity Spike:** Detected using Statistical Z-scores (3σ thresholds) on transaction frequency over a 7-day rolling window compared to historical baselines.
3. **Rapid Movement (Layering):** Detected by calculating tight time deltas between large incoming deposits and subsequent wire transfers out.
4. **Geographic Risk:** Flagged by calculating the percentage of a customer's transaction volume routed to high-risk FATF jurisdictions (e.g., KY, PA, VG, MM, IR).
5. **Dormant Activation:** Detected via gap analysis—flagging accounts with zero activity for 90+ days that suddenly execute massive wire transfers.
6. **Round-Tripping (Circular Flows):** Detected using NetworkX simple cycle detection (`circular_flow_count`) to find A -> B -> C -> A money trails. 

## 6. Dynamic Date Handling (The "Today" Anchor)
**Decision:** Anchor relative dates to dataset boundaries, not the system clock.
- **The Problem:** If a user queries "Show me activity in the last 30 days" on a static dataset from 2022, a naive `datetime.now()` anchor will yield zero results.
- **The Solution:** The Orchestrator calculates `df['timestamp'].max()` at startup. The `query_parser.py` uses this dynamic max date as the "Today" anchor. All relative queries are safely resolved against the actual data's timeline.

## 7. Human-In-The-Loop (HITL) Feedback Loop
**Decision:** Implement non-blocking, asynchronous feedback mechanisms.
- **The Feature:** Every flagged entity card features a "Confirm" and "FP" (False Positive) button. 
- **The Execution:** Clicking these buttons fires an asynchronous JavaScript `fetch()` request to `POST /api/feedback`. 
- **State Preservation:** To ensure the user doesn't lose their current analysis, the page *does not reload*. The DOM is manipulated locally to replace the buttons with a green checkmark or red 'X'. The backend appends the decision directly to `data/feedback.csv`, establishing an audit trail and paving the way for future Reinforcement Learning (RLHF).

## 8. Frontend UI / UX Aesthetics
**Decision:** Optimize for "Wow Factor" and Enterprise-Grade Polish.
- **Design Language:** Dark mode, glassmorphism (translucent panels with background blur), and strict adherence to a curated financial color palette (emerald greens, crimson reds, slate grays).
- **Typography & Iconography:** Utilizes professional sans-serif fonts and the `Lucide` icon library for crisp, scalable vector icons.
- **Agentic Visualization:** The `execution_trace` is rendered as an animated, monospace terminal log. Tools appear sequentially with a 350ms staggered delay, mimicking the visual feel of a real-time hacking/terminal interface to visually demonstrate the agent's autonomous reasoning.
