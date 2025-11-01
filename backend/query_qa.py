"""
query_qa_compatible.py

A Retrieval + LLM loop that DOES NOT import RetrievalQA.
Works reliably across LangChain versions by building the chain manually.
"""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
import textwrap

def load_vector_store(index_dir="faiss_hr_policy_index"):
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # load_local signature: (index_path, embeddings=..., allow_dangerous_deserialization=True)
    db = FAISS.load_local(
        index_dir,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    return db, embedding_model

def build_prompt_from_docs(docs, question, max_chars_per_doc=1200, max_total_chars=6000):
    """
    Build a concise prompt by taking the first N chars from each retrieved doc.
    Limits the total prompt size to avoid OOM / very long inputs.
    """
    parts = []
    total = 0
    for d in docs:
        snippet = (d.page_content or "").strip()
        if not snippet:
            continue
        snippet = snippet.replace("\n", " ").strip()
        snippet = snippet[:max_chars_per_doc].strip()
        source = d.metadata.get("source", d.metadata.get("filename", "unknown"))
        part = f"Source: {source}\n{snippet}"
        parts.append(part)
        total += len(part)
        if total > max_total_chars:
            break

    context = "\n\n---\n\n".join(parts)
    prompt = textwrap.dedent(f"""
    You are a helpful HR assistant. Use the document snippets below to answer the question.
    If the answer cannot be found in the provided snippets, respond: "I don't see relevant policy text in the documents."

    Context:
    {context}

    Question:
    {question}

    Provide:
    1) A concise direct answer (2-4 sentences).
    2) A bullet list of the sources (filenames) you used (if any).
    3) If you couldn't find the answer, say you didn't find it and suggest keywords to search.
    """).strip()
    return prompt

def main():
    # 1) Load index and create retriever
    db, embeddings = load_vector_store("faiss_hr_policy_index")
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # 2) Initialize local Ollama model via LangChain community wrapper
    #    (If you prefer langchain-ollama package, change import / usage accordingly)
    llm = Ollama(model="gemma3:1b", base_url="http://192.168.31.152:11434")  # change to the model you have (gemma, llama3, etc.)

    print("HR RAG CLI (compatible imports). Type 'exit' to quit.\n")
    while True:
        query = input("Ask HR policy question: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        # 3) Retrieve relevant docs (version-safe)
        try:
            docs = retriever.invoke(query)
        except Exception:
            try:
                docs = retriever.get_relevant_documents(query)
            except Exception:
                docs = retriever._get_relevant_documents(query)

        if not docs:
            print("No documents found in index.")
            continue


        # 4) Build prompt and call LLM
        prompt = build_prompt_from_docs(docs, query)
        try:
            # Call LLM: LangChain LLMs expose __call__, return can be a string
            answer = llm(prompt)
        except Exception as e:
            # If Ollama wrapper uses invoke or generate in your installed version, try fallback
            try:
                answer = llm.invoke(prompt)  # some wrappers use invoke
            except Exception:
                # last resort: show error to user
                print("LLM call failed:", repr(e))
                raise

        # 5) Print answer and sources
        print("\n=== Answer ===\n")
        # If answer is a dict-like object (varies by wrapper), handle gracefully
        if isinstance(answer, dict):
            # Many newer wrappers return {"result": "..."}
            out = answer.get("result") or answer.get("text") or str(answer)
            print(out)
        else:
            print(answer)

        print("\n--- Sources ---")
        for d in docs:
            src = d.metadata.get("source", d.metadata.get("filename", "unknown"))
            excerpt = (d.page_content or "").strip()[:200].replace("\n", " ")
            print(f"- {src}: {excerpt}...")
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
