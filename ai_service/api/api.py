import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import time
from utils.latency import metrics_ctx
from retrieval import rag_chain

app = FastAPI(title="Legally RAG API", version="1.0")

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = []

class SourceDocument(BaseModel):
    page_content: str
    metadata: dict

class AnalysisRequest(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    result: str

class ChatResponse(BaseModel):
    result: str
    source_documents: List[dict]
    trace_report: Optional[Dict[str, Any]] = None

import numpy as np

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj

@app.post("/api/v1/internal-chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    metrics_ctx.set({})
    x_trace_id = request.headers.get("X-Trace-ID", f"trace_{int(time.time())}")
    start_time = time.perf_counter()
    try:
        # Invoke the RAG chain
        response = rag_chain.invoke_qa(body.query, history=body.history)
        
        # Format the response
        result = response.get("result", "")
        source_docs = []
        
        for doc in response.get("source_documents", []):
            if hasattr(doc, "metadata"):
                # Clean metadata of numpy types
                metadata = convert_numpy_types(doc.metadata)
                source_docs.append({
                    "page_content": doc.page_content,
                    "metadata": metadata
                })
            else:
                # Handle case where it might be a dict already or something else
                source_docs.append(convert_numpy_types(doc))

        python_rag_total = int((time.perf_counter() - start_time) * 1000)
        metrics = metrics_ctx.get() or {}
        
        trace_report = {
            "metadata": {
                "id": x_trace_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "metrics_ms": {
                "python_rag_total": python_rag_total,
                "breakdown": metrics
            }
        }

        return ChatResponse(
            result=result,
            source_documents=source_docs,
            trace_report=trace_report
        )
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    try:
        # Call the analysis function from rag_chain
        result = rag_chain.analyze_text(request.text)
        return AnalysisResponse(result=result)
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/stats")
async def get_stats():
    try:
        # Get stats from Pinecone index via LangChain vectorstore
        # Note: This depends on the specific vectorstore implementation
        # For Pinecone, we can access the index directly
        stats = {
            "total_vectors": 0,
            "index_dimension": 0,
            "models": {
                "embedding": "multilingual-e5-large",
                "reranker": "BAAI/bge-reranker-v2-m3" if rag_chain.config.USE_RERANKER else "None"
            }
        }
        
        if hasattr(rag_chain, "_vector_store"):
            # Try to get Pinecone index stats
            try:
                index_stats = rag_chain._vector_store.get_pinecone_index().describe_index_stats()
                stats["total_vectors"] = index_stats.get("total_vector_count", 0)
                stats["index_dimension"] = index_stats.get("dimension", 0)
            except Exception as e:
                print(f"Failed to get Pinecone stats: {e}")
                # Fallback: maybe just return 0 or cached value
        
        return stats
    except Exception as e:
        print(f"Error fetching stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/generate-eval-data")
async def generate_eval_data(request: ChatRequest):
    try:
        # We reuse ChatRequest (query: str) for simpler reuse
        response = rag_chain.invoke_qa(request.query)
        
        result = response.get("result", "")
        chunks = [doc.page_content for doc in response.get("source_documents", [])]
        articles = []
        for doc in response.get("source_documents", []):
            code = doc.metadata.get("code_ru", "Неизвестно")
            article = doc.metadata.get("article_number", "")
            articles.append(f"{code} ст.{article}" if article else code)

        return {
            "answer": result,
            "chunks": chunks,
            "articles": articles
        }
    except Exception as e:
        print(f"Error generating eval data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
