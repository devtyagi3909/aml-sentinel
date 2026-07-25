"""
AML Sentinel — FastAPI Application Entry Point

Serves the agent API and the frontend dashboard.
Provides REST endpoints for query processing and WebSocket for streaming.
"""

import os
import sys
import json
import logging
import time
import asyncio
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from config import HOST, PORT, DEBUG, DATA_DIR, GOOGLE_API_KEY, LLM_PROVIDER, LLM_MODEL
from models.schemas import QueryRequest, AgentResponse, DatasetInfo
from agent.orchestrator import AMLSentinelAgent
from utils.data_loader import DataLoader


# ── Global State ──
agent: AMLSentinelAgent = None
data_loader: DataLoader = None


import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("aml_sentinel")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent and load data on startup."""
    logger.info("AML Sentinel initializing...")
    
    # Load default data
    data_dir = Path(__file__).resolve().parent.parent / "data"
    transactions_path = str(data_dir / "demo_transactions.csv")
    customers_path = str(data_dir / "sample_customers.csv")
    
    # Initialize data loader
    global agent, data_loader
    data_loader = DataLoader()
    
    # Check if data exists, generate if not
    txn_path = Path(transactions_path)
    cust_path = Path(customers_path)
    
    if not txn_path.exists():
        logger.info("Generating synthetic dataset...")
        gen_script = DATA_DIR / "generate_synthetic_data.py"
        if gen_script.exists():
            os.system(f"{sys.executable} {gen_script} --output-dir {DATA_DIR}")
        else:
            logger.warning("No data found and generator script missing. Please run generate_synthetic_data.py first.")
    
    # Load data
    try:
        data_loader.load(
            transactions_path=str(txn_path),
            customers_path=str(cust_path) if cust_path.exists() else None
        )
        logger.info(f"Loaded {len(data_loader.transactions):,} transactions, "
              f"{len(data_loader.customers):,} customers")
    except Exception as e:
        logger.error(f"Data loading error: {e}")
        logger.error("Generate data first: python data/generate_synthetic_data.py")
    
    # Initialize LLM (optional)
    llm = None
    if GOOGLE_API_KEY and LLM_PROVIDER == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL,
                google_api_key=GOOGLE_API_KEY,
                temperature=0,
            )
            logger.info(f"LLM initialized: {LLM_MODEL}")
        except ImportError:
            logger.warning("langchain-google-genai not installed. Using keyword-based query parsing.")
        except Exception as e:
            logger.warning(f"LLM initialization failed: {e}. Using keyword-based query parsing.")
    else:
        logger.info("No LLM API key configured. Using keyword-based query parsing.")
    
    # Initialize agent
    agent = AMLSentinelAgent(llm=llm)
    if data_loader.transactions is not None:
        agent.load_data(data_loader.transactions, data_loader.customers)
    
    logger.info("AML Sentinel startup sequence complete.")
    
    yield
    
    logger.info("AML Sentinel shutting down...")


# ── FastAPI App ──
app = FastAPI(
    title="AML Sentinel",
    description="AI-Powered Suspicious Activity Detection Agent",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Endpoints ──

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent_ready": agent is not None,
        "data_loaded": data_loader is not None and data_loader.transactions is not None,
        "transactions_count": len(data_loader.transactions) if data_loader and data_loader.transactions is not None else 0,
        "customers_count": len(data_loader.customers) if data_loader and data_loader.customers is not None else 0,
    }


@app.get("/api/dataset/info")
async def dataset_info():
    """Get dataset metadata."""
    if data_loader is None or data_loader.transactions is None:
        raise HTTPException(status_code=404, detail="No dataset loaded")
    
    info = data_loader.get_dataset_info()
    return info


@app.post("/api/query")
async def process_query(request: QueryRequest):
    """
    Process a natural language query through the AML agent.
    
    The agent will:
    1. Parse the query to extract intent and filters
    2. Build a dynamic execution plan
    3. Execute only the necessary tools
    4. Return structured results with explanations
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        response = await agent.process_query(request.query)
        return response.model_dump()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    """
    WebSocket endpoint for streaming query responses.
    Sends tool execution updates in real-time.
    """
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            query = data.get("query", "")
            
            if not query:
                await websocket.send_json({"error": "Empty query"})
                continue
            
            # Send acknowledgment
            await websocket.send_json({
                "type": "ack",
                "message": f"Processing: {query}"
            })
            
            # Process query
            try:
                response = await agent.process_query(query)
                
                # Send execution trace step by step
                for step in response.execution_trace:
                    await websocket.send_json({
                        "type": "trace",
                        "step": step.model_dump()
                    })
                    await asyncio.sleep(0.1)  # Small delay for animation
                
                # Send final response
                await websocket.send_json({
                    "type": "result",
                    "data": response.model_dump()
                })
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
    
    except WebSocketDisconnect:
        pass


class FeedbackRequest(BaseModel):
    entity_id: str
    decision: str

@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Save human-in-the-loop feedback to CSV without reloading."""
    feedback_file = Path(__file__).resolve().parent.parent / "data" / "feedback.csv"
    
    # Create file with headers if it doesn't exist
    if not feedback_file.exists():
        feedback_file.parent.mkdir(exist_ok=True, parents=True)
        with open(feedback_file, "w") as f:
            f.write("timestamp,entity_id,decision\n")
            
    # Append feedback
    with open(feedback_file, "a") as f:
        f.write(f"{datetime.now().isoformat()},{request.entity_id},{request.decision}\n")
        
    return {"status": "success", "message": "Feedback recorded"}

@app.get("/api/feedback/stats")
async def get_feedback_stats():
    """Get the current count of Confirm vs False Positive decisions."""
    feedback_file = Path(__file__).resolve().parent.parent / "data" / "feedback.csv"
    stats = {"confirmed": 0, "false_positives": 0}
    
    if feedback_file.exists():
        try:
            df = pd.read_csv(feedback_file)
            stats["confirmed"] = int((df["decision"] == "Confirm").sum())
            stats["false_positives"] = int((df["decision"] == "False Positive").sum())
        except Exception as e:
            logger.error(f"Error reading feedback stats: {e}")
            
    return stats


# ── Serve Frontend ──
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if FRONTEND_DIR.exists():
    @app.get("/")
    async def serve_frontend():
        """Serve the main dashboard page."""
        return FileResponse(FRONTEND_DIR / "index.html")
    
    # Mount static files
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    
    # Serve CSS and JS directly
    @app.get("/style.css")
    async def serve_css():
        return FileResponse(FRONTEND_DIR / "style.css", media_type="text/css")
    
    @app.get("/app.js")
    async def serve_js():
        return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")


# ── Run ──
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
    )
