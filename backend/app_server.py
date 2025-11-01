"""
rag_api.py — FastAPI wrapper for your HR RAG + LLM assistant
Uses FAISS + SentenceTransformer embeddings for HR document lookup,
and Ollama for both RAG and general questions.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
import os, textwrap

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.31.152:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:1b")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_hr_policy_index")

# -------------------------------------------------------------------
# INITIALIZE MODELS
# -------------------------------------------------------------------
app = FastAPI(title="HR Policy Assistant API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embeddings and FAISS vector store
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
retriever = None
if os.path.exists(FAISS_INDEX_PATH):
    db = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Initialize Ollama LLM
llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)

# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
HR_KEYWORDS = [
    "leave", "policy", "salary", "holiday", "attendance", "benefit", "bonus",
    "probation", "notice", "transfer", "gratuity", "vacation"
]

def is_hr_query(query: str) -> bool:
    query_lower = query.lower()
    return any(word in query_lower for word in HR_KEYWORDS)

def build_prompt_from_docs(docs, question, max_chars_per_doc=1200, max_total_chars=6000):
    """Constructs prompt from retrieved documents."""
    parts = []
    total = 0
    for d in docs:
        snippet = (d.page_content or "").strip()
        if not snippet:
            continue
        snippet = snippet.replace("\n", " ").strip()
        snippet = snippet[:max_chars_per_doc]
        source = d.metadata.get("source", d.metadata.get("filename", "unknown"))
        parts.append(f"Source: {source}\n{snippet}")
        total += len(snippet)
        if total > max_total_chars:
            break

    context = "\n\n---\n\n".join(parts)
    prompt = textwrap.dedent(f"""
    You are a helpful HR assistant. Use the document snippets below to answer the question.
    If the answer cannot be found in the provided snippets, say:
    "I don't see relevant policy text in the documents."

    Context:
    {context}

    Question:
    {question}

    Provide:
    1) A concise answer (2-4 sentences).
    2) A bullet list of the sources (filenames) used, if any.
    3) If not found, say you didn't find it and suggest search keywords.
    """).strip()
    return prompt

def call_llm(prompt: str):
    """Safely call the LLM with backward compatibility for invoke/call."""
    try:
        return llm(prompt)
    except Exception:
        try:
            return llm.invoke(prompt)
        except Exception as e:
            return f"Error calling LLM: {repr(e)}"

# -------------------------------------------------------------------
# API MODELS
# -------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    mode: str
    answer: str

# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------
@app.post("/query", response_model=QueryResponse)
async def handle_query(req: QueryRequest):
    query = req.query.strip()
    if not query:
        return QueryResponse(mode="error", answer="Empty query provided.")

    # General vs HR mode
    if is_hr_query(query) and retriever:
        # Use similarity search
        try:
            docs = retriever.invoke(query)
        except Exception:
            try:
                docs = retriever.get_relevant_documents(query)
            except Exception:
                docs = retriever._get_relevant_documents(query)

        if not docs:
            return QueryResponse(mode="RAG", answer="No relevant documents found in index.")

        prompt = build_prompt_from_docs(docs, query)
        answer = call_llm(prompt)
        return QueryResponse(mode="RAG", answer=answer)
    else:
        # Direct LLM
        answer = call_llm(query)
        return QueryResponse(mode="Direct LLM", answer=answer)

@app.get("/")
def root():
    return {"status": "ok", "mode": "RAG + Direct LLM", "model": LLM_MODEL, "ollama": OLLAMA_BASE_URL}
