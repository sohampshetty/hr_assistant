"""
app_server.py — HR Assistant API
Now includes intent detection for routing (leave_balance / policy_query / general)
"""

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
import os, textwrap, requests
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from typing import Optional
from sentence_transformers import SentenceTransformer, util
from bson import ObjectId
from fastapi import HTTPException




# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.31.152:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:1b")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_hr_policy_index")
LEAVE_BALANCE_API = os.getenv("LEAVE_BALANCE_API", "http://localhost:8080/user")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://192.168.31.152:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "hr_assistant")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "users")

mongo_client: Optional[AsyncIOMotorClient] = None
users_collection = None


# -------------------------------------------------------------------
# INITIALIZE MODELS
# -------------------------------------------------------------------
app = FastAPI(title="HR Policy Assistant API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_db_client():
    global mongo_client, users_collection
    mongo_client = AsyncIOMotorClient(MONGO_URI)
    db = mongo_client[MONGO_DB_NAME]
    users_collection = db[MONGO_COLLECTION]
    print("✅ Connected to MongoDB")

@app.on_event("shutdown")
async def shutdown_db_client():
    if mongo_client:
        mongo_client.close()
        print("🛑 MongoDB connection closed")

async def get_user_details_from_db(user_id: str):
    """
    Fetch a user's leave details from MongoDB.
    Returns a dict similar to what your old /user/{user_id} API returned.
    """
    try:
        # Handle both ObjectId and string user_id fields
        query = {"_id": ObjectId(user_id)} if ObjectId.is_valid(user_id) else {"user_id": user_id}
        user = await users_collection.find_one(query)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return {
            "id": str(user.get("_id", "")),
            "name": user.get("username", "Unknown"),
            "remaining_leaves": user.get("leave_balance", 0),
            "total_leaves": user.get("total_leaves", 100)
        }
    except Exception as e:
        print(f"❌ Error fetching user from MongoDB: {e}")
        raise HTTPException(status_code=500, detail="Database error")

# Embedding model for FAISS + intent detection
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
intent_model = SentenceTransformer("all-MiniLM-L6-v2")

retriever = None
if os.path.exists(FAISS_INDEX_PATH):
    db = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Ollama LLM
llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)

# -------------------------------------------------------------------
# INTENT DETECTION
# -------------------------------------------------------------------
INTENT_EXAMPLES = {
    "leave_balance": [
        "how many leaves left",
        "check my leave balance",
        "remaining leave days",
        "tell me my leave balance",
        "how many paid leaves do I have",
        "leave summary",
        "casual leave left"
    ],
    "policy_query": [
        "what is maternity policy",
        "notice period",
        "transfer rules",
        "bonus policy",
        "gratuity policy",
        "holiday list",
        "probation policy"
    ],
    "general": [
        "hi",
        "hello",
        "thank you",
        "who are you",
        "what can you do"
    ]
}

intent_embeddings = {
    k: intent_model.encode(v, convert_to_tensor=True)
    for k, v in INTENT_EXAMPLES.items()
}

def detect_intent_embedding(query: str) -> str:
    query_emb = intent_model.encode(query, convert_to_tensor=True)
    scores = {
        intent: util.cos_sim(query_emb, emb).max().item()
        for intent, emb in intent_embeddings.items()
    }
    best_intent = max(scores, key=scores.get)
    confidence = scores[best_intent]
    if confidence < 0.55:
        return "unknown"
    return best_intent


def detect_intent_llm(query: str) -> str:
    """Ask LLM for fallback intent classification"""
    prompt = textwrap.dedent(f"""
    You are an intent classifier for an HR chatbot.
    Given a user query, respond with ONLY one word:
    - leave_balance → if asking about remaining or total leaves
    - policy_query → if asking about HR policy (maternity, notice, holiday, bonus)
    - general → if unrelated or casual chat
    Query: {query}
    Intent:
    """)
    try:
        response = llm(prompt).strip().lower()
    except Exception:
        response = "general"
    if response not in ["leave_balance", "policy_query", "general"]:
        response = "general"
    return response

def detect_intent_llm(query: str) -> str:
    """
    Use the LLM itself to decide user intent dynamically.
    """
    prompt = f"""
    You are an intent classifier for an HR chatbot. 
    Classify the user's intent into one of these categories:
    1) leave_balance — when the user asks about remaining leaves, how many leaves left, etc.
    2) policy_query — when the user asks about HR policies (maternity, leave rules, notice period, etc.)
    3) general — for greetings, small talk, or unrelated topics.

    User query: "{query}"

    Return only one word: leave_balance, policy_query, or general.
    """

    try:
        result = llm.invoke(prompt)
        text = str(result).lower()
        if "leave_balance" in text:
            return "leave_balance"
        elif "policy_query" in text:
            return "policy_query"
        else:
            return "general"
    except Exception:
        return "general"

def detect_intent(query: str) -> str:
    """Hybrid intent detection with embedding + LLM fallback"""
    intent = detect_intent_embedding(query)
    if intent == "unknown":
        intent = detect_intent_llm(query)
    return intent


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def build_prompt_from_docs(docs, question, max_chars_per_doc=1200, max_total_chars=6000):
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
    user_id: Optional[str] = None

class QueryResponse(BaseModel):
    mode: str
    intent: str
    answer: str


# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------
@app.post("/query", response_model=QueryResponse)
async def handle_query(req: QueryRequest):
    query = req.query.strip()
    if not query:
        return QueryResponse(mode="error", intent="none", answer="Empty query provided.")

    intent = detect_intent_llm(query)
    print(f"🧭 Detected intent: {intent}")

    # --- Leave Balance ---
    if intent == "leave_balance":
        if not req.user_id:
            return QueryResponse(mode="API", intent=intent, answer="User ID is required to check leave balance.")
        
        try:
            user_data = await get_user_details_from_db(req.user_id)
        except HTTPException as e:
            return QueryResponse(mode="API", intent=intent, answer=e.detail)

        # Build a prompt using DB data and user query
        prompt = f"""
        The user asked: "{req.query}"
        According to the HR database:
        - Name: {user_data['name']}
        - Remaining Leaves: {user_data['remaining_leaves']}
        - Total Leaves: {user_data['total_leaves']}

        Generate a clear and friendly HR chatbot response that explains their leave balance.
        Example: "You currently have 8 remaining out of 20 total leaves."
        """

        answer = call_llm(prompt)
        return QueryResponse(mode="LLM+DB", intent=intent, answer=answer)

    elif intent == "policy_query" and retriever:
        try:
            docs = retriever.invoke(query)
        except Exception:
            docs = retriever.get_relevant_documents(query)
        if not docs:
            return QueryResponse(mode="RAG", intent=intent, answer="No relevant HR documents found.")
        prompt = build_prompt_from_docs(docs, query)
        answer = call_llm(prompt)
        return QueryResponse(mode="RAG", intent=intent, answer=answer)

    else:
        answer = call_llm(query)
        return QueryResponse(mode="Direct LLM", intent=intent, answer=answer)


@app.get("/")
def root():
    return {
        "status": "ok",
        "mode": "RAG + Direct LLM + Intent Detection",
        "model": LLM_MODEL,
        "ollama": OLLAMA_BASE_URL
    }
