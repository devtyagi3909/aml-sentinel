# 🛡️ AML Sentinel — AI-Powered Suspicious Activity Detection

An autonomous, agentic AI system for Anti-Money Laundering (AML) compliance. AML Sentinel dynamically parses natural language queries, constructs intelligent execution plans, and orchestrates specialized analysis tools to detect money laundering patterns — providing explainable risk assessments with actionable escalation recommendations.

## 🎯 Problem Statement

Financial institutions face regulatory mandates (FinCEN, FATF) to detect and report suspicious activities. Traditional rule-based AML systems generate excessive false positives while sophisticated laundering techniques evade detection. AML Sentinel addresses this by combining machine learning, statistical analysis, and rule-based detection in an agentic architecture that adapts its analysis to each specific query.

## 🏗️ Solution Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    🖥️ Web Dashboard                           │
│         Chat Interface │ Visualizations │ Risk Dashboard      │
└───────────────────────────┬──────────────────────────────────┘
                            │ REST API / WebSocket
┌───────────────────────────┴──────────────────────────────────┐
│                    ⚡ FastAPI Backend                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │Query Parser  │→│Dynamic Planner│→│Agent Orchestrator   │  │
│  │(NLP/Keyword) │  │(Intent→Tools) │  │(Tool Coordination) │  │
│  └─────────────┘  └──────────────┘  └────────┬───────────┘  │
│                                               │              │
│  ┌──────┐ ┌────────┐ ┌─────────┐ ┌──────┐ ┌─────────────┐  │
│  │ EDA  │ │Feature │ │Anomaly  │ │ Risk │ │ Explanation  │  │
│  │ Tool │ │Engineer│ │Detector │ │Class.│ │   Engine     │  │
│  └──────┘ └────────┘ └─────────┘ └──────┘ └─────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### What Makes It "Agentic"

AML Sentinel is **not a fixed sequential pipeline**. The agent:

1. **Parses** natural language queries to extract intent, filters, entities, and target patterns
2. **Plans** dynamically — different queries invoke different tool combinations:
   - `"Analyze the dataset"` → EDA → Feature Engineering → Anomaly Detection → Risk Classification → Explanation (5 tools)
   - `"Is customer C-4521 suspicious?"` → Feature Engineering → Anomaly Detection → Risk Classification → Explanation (4 tools, EDA skipped)
   - `"Which customers made 10+ cash deposits under $10,000?"` → Feature Engineering only (ML skipped)
   - `"Show me data distributions"` → EDA only
3. **Executes** only the necessary tools on the relevant data subset
4. **Explains** every flag with specific evidence tied to the original query

## 🔍 Detection Capabilities

### AML Patterns Detected

| Pattern | Description | Detection Method |
|---------|-------------|-----------------|
| **Structuring/Smurfing** | Cash deposits $8K-$9.99K below CTR threshold | Rule-based + clustering |
| **Rapid Movement** | Large deposit → immediate wire transfer | Time-window analysis |
| **Velocity Spike** | Sudden 5-10x increase in transaction frequency | Statistical deviation |
| **Dormant Activation** | 90+ day inactive account suddenly active | Gap analysis |
| **Geographic Risk** | Transactions to FATF grey/black list countries | Jurisdiction rules |
| **Round-Trip** | Circular fund movement (A→B→C→A) | Graph pattern matching |

### Hybrid Anomaly Detection

- **Isolation Forest** (ML): Unsupervised anomaly detection on feature vectors
- **Z-Score Analysis** (Statistical): Flags transactions >3σ from customer baseline
- **Rule-Based Patterns**: Hard-coded AML rules with regulatory thresholds
- **Ensemble Scoring**: Weighted combination (Rules: 45%, IF: 30%, Z-Score: 25%)

## 📊 Dataset

### Source & Generation

The dataset is **synthetically generated** using Python's `Faker` and `NumPy` libraries with deliberate AML pattern injection. This approach was chosen because:

- Real AML data is proprietary and highly regulated
- Synthetic data allows controlled injection of known patterns for validation
- Ground truth labels enable precision/recall measurement

### Schema

**Transactions** (~50,000 records): `transaction_id`, `customer_id`, `timestamp`, `amount`, `transaction_type`, `channel`, `counterparty_id`, `counterparty_country`, `currency`, `account_balance`, `is_cash`, `merchant_category`, `ground_truth_flag`, `ground_truth_pattern`

**Customers** (~5,000 records): `customer_id`, `name`, `age`, `occupation`, `account_type`, `account_open_date`, `country`, `risk_category`, `pep_flag`, `kyc_status`

### Data Generation Logic

- Normal transactions: 80% small retail ($5-$500), 20% medium ($500-$5,000)
- Suspicious patterns: ~2-3% of total volume, injected across 6 pattern types
- See `data/generate_synthetic_data.py` for complete generation logic with `--verify` flag

## 🛠️ Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Agent Framework | Python + Custom Orchestrator | 3.10+ |
| Backend | FastAPI + Uvicorn | 0.115+ |
| Anomaly Detection | scikit-learn (Isolation Forest) | 1.5+ |
| Data Processing | Pandas + NumPy | 2.2+ |
| Visualization | Matplotlib (backend) + Chart.js (frontend) | - |
| Frontend | Vanilla HTML/CSS/JS | - |
| Synthetic Data | Faker + NumPy | - |
| LLM (optional) | Google Gemini (free tier) | 2.0-flash |

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- pip

### Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd aml-sentinel

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate synthetic dataset
python data/generate_synthetic_data.py --output-dir data/ --verify

# 5. (Optional) Configure LLM for enhanced query parsing
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# 6. Start the application
cd backend
python main.py
```

The dashboard will be available at **http://localhost:8000**

### Without LLM (Works Great!)

The system works fully without an LLM API key. It uses intelligent keyword-based query parsing that handles all the example queries. LLM integration enhances natural language understanding but is not required.

## 💡 Usage

### Example Queries

| Query | What the Agent Does |
|-------|-------------------|
| `"Analyze this dataset for suspicious activity"` | Full pipeline: EDA → Features → Anomaly Detection → Risk → Explain |
| `"Find structuring patterns in the last 30 days"` | Time-filtered, structuring-focused analysis (skips full EDA) |
| `"Which customers made 10+ cash deposits under $10,000?"` | Direct aggregation — ML not needed |
| `"Is customer C-4521 suspicious?"` | Single-entity analysis with risk explanation |
| `"Show high-risk wire transfers"` | Pattern search filtered to wire transfers |
| `"Compare customers C-1001 and C-2045"` | Side-by-side comparison |

### API Endpoints

- `POST /api/query` — Submit natural language query
- `GET /api/health` — Health check
- `GET /api/dataset/info` — Dataset metadata
- `WS /ws/query` — WebSocket streaming

## 📋 External Tools & AI Disclosure

- **Google Gemini API** (optional): Used for enhanced query parsing. The system works fully without it.
- **scikit-learn**: Isolation Forest for anomaly detection
- **Faker**: Synthetic data generation
- **Agentic coding tools**: Used during development as permitted by hackathon rules

## 📁 Project Structure

```
aml-sentinel/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── generate_synthetic_data.py    # Synthetic dataset generator
│   ├── sample_transactions.csv       # Generated transactions
│   └── sample_customers.csv          # Generated customers
├── backend/
│   ├── main.py                       # FastAPI entry point
│   ├── config.py                     # Configuration
│   ├── agent/
│   │   ├── orchestrator.py           # Agent brain (dynamic routing)
│   │   ├── query_parser.py           # NLP query understanding
│   │   ├── planner.py                # Dynamic execution planning
│   │   └── state.py                  # Agent state management
│   ├── tools/
│   │   ├── eda_tool.py               # Exploratory data analysis
│   │   ├── feature_engineering.py    # AML feature creation
│   │   ├── anomaly_detection.py      # ML + rule-based detection
│   │   ├── risk_classification.py    # Risk scoring & categorization
│   │   └── explanation_engine.py     # Natural language explanations
│   ├── models/
│   │   ├── schemas.py                # Pydantic models
│   │   └── enums.py                  # Enumerations
│   └── utils/
│       └── data_loader.py            # Data loading & caching
└── frontend/
    ├── index.html                    # Dashboard layout
    ├── style.css                     # Dark theme design system
    └── app.js                        # Interactive UI logic
```

## 📄 License

This project was created for a hackathon. All code is original and created during the hackathon window.
