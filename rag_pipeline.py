# rag_pipeline.py
import os
import tempfile
import threading
import json
from typing import List, Tuple, Optional, Dict
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from groq import Groq
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from pydantic import Field, PrivateAttr

# Load env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not found. Please set it in your .env file.")

# Config / constants
DB_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # faster, good for interactive use
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RE_RANK_TOP_K = 20  # retrieve this many then re-rank to top_n
RE_RANK_FINAL = 4
SUMMARIZE_BATCH = 8  # number of chunks per intermediate summarization
LOG_FILE = "interactions_log.jsonl"

# ---- Utilities ----
def _ensure_db_dir():
    os.makedirs(DB_DIR, exist_ok=True)

def _log_interaction(record: dict):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass  # logging must not break flow

# ---- Embeddings & LLM classes ----
class GroqLLM(LLM):
    """
    Minimal wrapper around Groq chat completions. Uses synchronous calls.
    """
    model: str = Field(default="openai/gpt-oss-120b")  # swap if you want another Groq model
    _client: Groq = PrivateAttr()

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if api_key is None:
            raise ValueError("API key must be provided for Groq client.")
        self._client = Groq(api_key=api_key)

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_completion_tokens=1024,
            top_p=1,
            stop=stop,
            stream=False,
        )
        return response.choices[0].message.content.strip()

# ---- Global cached components (module-level) ----
# These are lazily initialized to avoid expensive work on import
_cached = {
    "embeddings": None,
    "vectordb": None,
    "retriever": None,
    "llm": None,
}

def init_components(force_reload: bool = False):
    """
    Initialize (or reuse) Embeddings, Chroma vectordb, retriever and Groq LLM client.
    Call this once at app start or after processing documents.
    """
    _ensure_db_dir()

    if _cached["embeddings"] is None or force_reload:
        _cached["embeddings"] = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)

    if _cached["vectordb"] is None or force_reload:
        # If database exists, load persisted Chroma, else keep None until process_pdfs created it
        try:
            _cached["vectordb"] = Chroma(persist_directory=DB_DIR, embedding_function=_cached["embeddings"])
        except Exception:
            _cached["vectordb"] = None

    if _cached["vectordb"] is not None:
        _cached["retriever"] = _cached["vectordb"].as_retriever(search_kwargs={"k": RE_RANK_TOP_K})
    else:
        _cached["retriever"] = None

    if _cached["llm"] is None or force_reload:
        _cached["llm"] = GroqLLM(api_key=GROQ_API_KEY)

    return _cached["embeddings"], _cached["vectordb"], _cached["retriever"], _cached["llm"]

# ---- Document processing / ingestion ----
def _load_and_split_pdf(tmp_path: str):
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    # attach simple metadata if not present (filename, page) — PyPDFLoader usually provides page metadata
    return chunks

def process_pdfs(uploaded_files: List) -> Tuple[bool, str]:
    """
    Process uploaded PDFs synchronously. Use process_pdfs_in_background for non-blocking.
    Returns (success, message).
    """
    if not uploaded_files:
        return False, "No files provided."

    _ensure_db_dir()
    local_chunks = []
    for uploaded_file in uploaded_files:
        # stream to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        try:
            chunks = _load_and_split_pdf(tmp_path)
            # add filename metadata for citation
            for c in chunks:
                if "metadata" not in c.metadata:
                    c.metadata = {"source": os.path.basename(uploaded_file.name)}
                else:
                    c.metadata["source"] = os.path.basename(uploaded_file.name)
            local_chunks.extend(chunks)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    if not local_chunks:
        return False, "No text extracted from provided PDFs."

    # instantiate embeddings and create Chroma DB from documents (persist)
    embeddings, _, _, _ = init_components(force_reload=True)
    vectordb = Chroma.from_documents(local_chunks, embedding=embeddings, persist_directory=DB_DIR)
    vectordb.persist()

    # reload cached components so retriever is available
    init_components(force_reload=True)
    return True, f"Processed {len(local_chunks)} chunks and persisted to vector DB."

def process_pdfs_in_background(uploaded_files: List, status_dict: dict):
    """
    Runs process_pdfs in a background thread and updates status_dict with {"status": "running"/"done"/"error", "msg": ...}
    """
    try:
        status_dict["status"] = "running"
        success, msg = process_pdfs(uploaded_files)
        status_dict["status"] = "done" if success else "error"
        status_dict["msg"] = msg
    except Exception as e:
        status_dict["status"] = "error"
        status_dict["msg"] = str(e)

# ---- Retrieval & reranking ----
import math

def _cosine_sim(a: List[float], b: List[float]) -> float:
    # safe cosine similarity
    dot = sum(x*y for x,y in zip(a,b))
    sa = math.sqrt(sum(x*x for x in a))
    sb = math.sqrt(sum(x*x for x in b))
    if sa == 0 or sb == 0: return 0.0
    return dot / (sa*sb)

def retrieve_and_rerank(query: str, top_k: int = RE_RANK_TOP_K, final_n: int = RE_RANK_FINAL) -> List[Dict]:
    """
    Retrieves top_k docs then re-ranks them using embeddings cosine similarity to the query embedding.
    Returns list of dicts: { "content": ..., "metadata": ..., "score": ... }
    """
    embeddings, vectordb, retriever, _ = init_components()
    if retriever is None:
        raise ValueError("No vector store available. Please process documents first.")

    # initial retrieval using retriever
    docs = retriever.get_relevant_documents(query)  # returns up to top_k as configured
    if not docs:
        return []

    # compute embeddings for query and each doc content
    try:
        query_emb = embeddings.embed_query(query)
    except Exception:
        # fallback: small wrapper if method name differs
        query_emb = embeddings.embed_documents([query])[0]

    doc_texts = [d.page_content for d in docs]
    try:
        doc_embs = embeddings.embed_documents(doc_texts)
    except Exception:
        # fallback single
        doc_embs = [embeddings.embed_query(t) for t in doc_texts]

    scored = []
    for d, emb in zip(docs, doc_embs):
        score = _cosine_sim(query_emb, emb)
        scored.append({"content": d.page_content, "metadata": getattr(d, "metadata", {}), "score": float(score)})

    # sort by score descending and return top final_n
    scored_sorted = sorted(scored, key=lambda x: x["score"], reverse=True)
    return scored_sorted[:final_n]

# ---- Summarization pipeline (hierarchical) ----
def _llm_summarize_text(llm: GroqLLM, text: str, prompt_heading: str = "Summarize the following text:") -> str:
    prompt = f"{prompt_heading}\n\n{text}\n\nProvide a concise but comprehensive summary."
    return llm(prompt)

def summarize_query(query: str) -> str:
    """
    Handles 'summarize' intent. Retrieves many chunks, groups them, summarizes per group,
    and then composes a final summary.
    """
    embeddings, vectordb, retriever, llm = init_components()
    if retriever is None:
        return "No documents indexed. Please upload and process documents first."

    # use the underlying vectordb search to retrieve more context (bypass retriever top_k)
    # We'll call the retriever but with a larger k if possible by creating an ad-hoc one:
    ad_hoc_retriever = vectordb.as_retriever(search_kwargs={"k": max(50, RE_RANK_TOP_K)})
    docs = ad_hoc_retriever.get_relevant_documents(query)
    if not docs:
        return "Couldn't find relevant content to summarize."

    # extract contents
    chunks = [d.page_content for d in docs]

    # group into batches and summarize each batch
    intermediate_summaries = []
    for i in range(0, len(chunks), SUMMARIZE_BATCH):
        batch_text = "\n\n".join(chunks[i:i+SUMMARIZE_BATCH])
        summary = _llm_summarize_text(llm, batch_text, prompt_heading="Summarize the following passage in 4-6 bullet points:")
        intermediate_summaries.append(summary)

    # combine intermediate summaries and create a final summary
    combined = "\n\n".join(intermediate_summaries)
    final = _llm_summarize_text(llm, combined, prompt_heading="Now create a final structured summary from the combined summaries. Include key points and headings where useful.")
    return final

# ---- Main QA function (routes intent -> response) ----
def detect_intent(query: str) -> str:
    qlower = query.lower()
    if any(k in qlower for k in ["summarize", "summarise", "summary", "summation", "short summary", "summarization"]):
        return "summarize"
    # add more intents later: "quiz", "translate", "explain", etc.
    return "qa"

def answer_query(query: str, return_sources: bool = True) -> Dict:
    """
    Main entrypoint. Returns a dictionary:
    {
        "answer": str,
        "sources": [ { "metadata": ..., "score": ... }, ... ],
        "intent": "qa" / "summarize",
        "error": None or str
    }
    """
    try:
        embeddings, vectordb, retriever, llm = init_components()
    except Exception as e:
        return {"answer": "", "sources": [], "intent": None, "error": f"Initialization error: {e}"}

    intent = detect_intent(query)

    try:
        if intent == "summarize":
            answer_text = summarize_query(query)
            sources = []  # summarization already pulled many docs; we could return top sources if needed
        else:
            # retrieval + rerank
            reranked = retrieve_and_rerank(query)
            if not reranked:
                return {"answer": "I couldn't find relevant content in the uploaded documents. Try re-uploading or processing documents.", "sources": [], "intent": intent, "error": None}

            # prepare context with citations
            context_blocks = []
            sources = []
            for idx, item in enumerate(reranked):
                meta = item.get("metadata", {}) or {}
                source_label = meta.get("source", f"chunk_{idx}")
                snippet = item["content"][:800]  # short snippet for the prompt
                context_blocks.append(f"[Source: {source_label} | score: {item['score']:.3f}]\n{snippet}")
                sources.append({"metadata": meta, "score": item["score"]})

            # Compose prompt for LLM
            context_text = "\n\n---\n\n".join(context_blocks)
            prompt = (
                "You are an expert tutor answering student queries using only the provided context. "
                "If the answer is not contained in the context, say you cannot find it in the documents. "
                "Cite sources inline using [Source: filename].\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {query}\n\n"
                "Answer concisely and provide references to the sources used."
            )
            answer_text = llm(prompt)

            # Logging
            _log_interaction({
                "query": query,
                "intent": intent,
                "answer": answer_text,
                "sources": sources,
            })

        return {"answer": answer_text, "sources": sources, "intent": intent, "error": None}
    except Exception as e:
        return {"answer": "", "sources": [], "intent": intent, "error": str(e)}
