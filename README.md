# AML Sentinel

A distributed, horizontally scalable system for Anti-Money Laundering (AML) compliance and anomaly detection in financial networks. AML Sentinel utilizes a dynamic routing architecture to parse query intent, construct execution graphs, and orchestrate specialized analysis modules for detecting structuring, layering, and cyclic obfuscation patterns.

## Problem Statement

Financial institutions operate under strict regulatory mandates (FinCEN, FATF) to detect and report suspicious activities. Traditional rule-based AML systems suffer from low precision and high false-positive rates, while sophisticated laundering techniques easily evade static thresholds. This system addresses these limitations by fusing unsupervised machine learning (Isolation Forest), statistical divergence (Z-Score distribution), graph theory (NetworkX centrality), and deterministic rule evaluation into a dynamic execution pipeline.

## System Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                    Web Dashboard                             │
│         Query Interface │ Data Viz │ Telemetry Dashboard     │
└───────────────────────────┬──────────────────────────────────┘
                            │ REST API
┌───────────────────────────┴──────────────────────────────────┐
│                    FastAPI Backend                           │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │Query Parser │→ │Graph Planner │→ │Pipeline Orchestrator│  │
│  │(Regex/AST)  │  │(DAG Builder) │  │(State Management)  │   │
│  └─────────────┘  └──────────────┘  └────────┬───────────┘   │
│                                              │               │
│  ┌──────┐ ┌────────┐ ┌─────────┐ ┌──────┐ ┌─────────────┐    │
│  │ EDA  │ │Feature │ │Anomaly  │ │ Risk │ │ Explanation │    │
│  │ Mod. │ │Engineer│ │Detector │ │Class.│ │ Engine      │    │
│  └──────┘ └────────┘ └─────────┘ └──────┘ └─────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### Dynamic Execution Pipeline

Unlike static ETL pipelines, the orchestrator:
1. **Parses** input constraints to extract filters, entity IDs, and target topological patterns.
2. **Plans** a Directed Acyclic Graph (DAG) for tool execution based on the request bounds.
3. **Executes** modules conditionally (e.g. bypassing ML inference for deterministic aggregation queries).
4. **Synthesizes** mathematical evidence into an explainable matrix.

## Detection Methodology

### Evaluated Topologies

| Topology | Description | Detection Mechanism |
|---------|-------------|-----------------|
| **Structuring** | Cash deposits immediately below CTR threshold | Deterministic window aggregation |
| **Rapid Movement** | High-velocity deposit-to-withdrawal sequence | Temporal delta analysis |
| **Velocity Spike** | Statistically significant transaction frequency deviation | Standard deviation thresholding |
| **Dormant Activation** | Reactivation post 90+ day latency | Time-series gap analysis |
| **Geographic Risk** | Capital flight to FATF high-risk jurisdictions | Cross-reference mapping |
| **Round-Trip Flow** | Closed-loop circular fund movement | Directed graph cycle detection |

### Hybrid Anomaly Detection

- **Unsupervised ML**: Scikit-Learn `IsolationForest` on multidimensional feature vectors.
- **Statistical**: Z-Score divergence tracking across customer historical baselines.
- **Graph Mathematics**: `NetworkX` PageRank and cycle-basis evaluation for network centrality.
- **Ensemble Synthesis**: Weighted coefficient scoring across all subsystems.

## Dataset Engineering

The system analyzes financial transactions using the **Kaggle IBM Transactions for Anti Money Laundering (AML)** dataset.
- For the live hackathon presentation and memory safety, we dynamically sample the first **50,000 transactions** via the `DataLoader`.
- The dataset features complex topologies including velocity spikes, geographic risk, and structured layering.
- Ground truth labels are preserved strictly for classification metric evaluation (Precision, Recall, F1).

## Technology Stack

| Subsystem | Technology |
|-----------|-----------|
| Orchestration | Python 3.10+ |
| Web Server | FastAPI, Uvicorn |
| Machine Learning | Scikit-Learn |
| Data Processing | Pandas, NumPy |
| Network Analysis | NetworkX |
| Interface | HTML5, CSS3, ES6 JavaScript |

## Installation & Deployment

### Local Environment

```bash
git clone <repository_url>
cd aml-sentinel
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python data/generate_synthetic_data.py --output-dir data/
cd backend
python main.py
```
Access the telemetry interface at `http://localhost:8000`.

### Cloud Deployment (Render.com)

The repository includes a standard `render.yaml` infrastructure-as-code configuration. Connect the repository to Render as a Blueprint to automatically provision the required compute environment and install all dependencies.

## Usage Documentation

Submit constraints via the REST API or Web UI. Supported query structures include:
- `Find circular round-trip laundering in wire transfers`
- `Analyze the dataset for rapid movement and dormant account activation`
- `Is customer C-3023 suspicious?`

### REST Interface

- `POST /api/query`: Submits parameterized query for pipeline evaluation.
- `GET /api/health`: Validates system state and memory allocation.
- `GET /api/dataset/info`: Returns corpus metadata schema.

## License
Provided strictly for hackathon evaluation purposes.
