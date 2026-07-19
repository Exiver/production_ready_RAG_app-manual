import io 
import json 
import time 
import os 
from datetime import datetime 

import faiss
import numpy as np
import pandas as pd 
import requests
import streamlit as st 
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder

st.set_page_config(page_title="RAG_Chatbot", page_icon="🚀📚", layout="wide")


embedding_model = "all-MiniLM-L6-v2"
cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
chunk_size_words = 220
chunk_overlap_words = 40
rerank_candedate_pool = 10

@st.cache_resource(show_spinner="Loading embedding model ...")
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(embedding_model)

@st.cache_resource(show_spinner="Loading cross encoder re-ranker .....")
def load_cross_encoder() -> CrossEncoder:
    return CrossEncoder(cross_encoder_model)

embedder = load_embedder()
cross_encoder = load_cross_encoder()

def load_pdf_pages(file_bytes: bytes) -> list[dict]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = " ".join((page.extract_text() or "").split())
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages

def chunk_pages(pages: list[dict], chunk_size: int = chunk_size_words, overlap: int = chunk_overlap_words) -> list[dict]:
    wird_page_pairs =  [(w, p["page"]) for p in pages for w in p["text"].split()]
    step = chunk_size - overlap
    chunk, chunk_id = [], 0
    for start in range(0, len(wird_page_pairs), step):
        window  = wird_page_pairs[start:start + chunk_size]
        if not window:
            break
        chunk.append({
            "chunk_id": chunk_id,
            "text": " ".join(w for w,_ in window),
            "page_start": window[0][1],
            "page_end": window[-1][1],
        })
        chunk_id +=1
        if start + chunk_size >= len(wird_page_pairs):
            break
    return chunk

def build_index(chunk: list[dict]):
    embeddings = embedder.encode(
        [c["text"] for c in chunk], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
    )
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

#---------------------------------------------------------
# Retrieveal 

def retrieve(query: str, index, chunk: list[dict], k: int = 3, use_re_rank: bool = True, pool: int= rerank_candedate_pool) -> list[dict]:
    q_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    n_fetch = max(pool, k) if use_re_rank else k
    scores, idxs = index.search(q_vec, n_fetch)
    candedates = [
        {"text": chunk[i]["text"],
         "page_start": chunk[i]["page_start"],
         "page_end": chunk[i]["page_end"],
         "score": float(s)}
         for s, i in zip(scores[0], idxs[0]) if i != -1
    ]

    if use_re_rank and candedates:
        pairs = [(query, c["text"]) for c in candedates]
        rerank_scores = cross_encoder.predict(pairs)
        for c, rs in zip(candedates, rerank_scores):
            c["re-rank_score"] = float(rs)
        candedates.sort(key= lambda c: c["re-rank_score"], reverse=True)
    return candedates[:k]

def build_prompt(query:str, retrieved_texts: list[dict]) -> str:
    context = "\n\n".join(retrieved_texts)
    return (
        'You are a careful assistant. Answer the question using ONLY the context below.\n'
        'If the answer is not contained in the context, say "I don\'t have enough information '
        'in the retrieved context to answer that." Do not use outside knowledge.\n\n'
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )

def call_ollama(prompt: str, model: str, url: str, temperature: float = 0.0) -> str:
    resp = requests.post(
        url,
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "stream": False, "options": {"temperature": temperature}},
              timeout=120,
    )
    resp.raise_for_status()
    text = resp.text.strip()
    full_content = ""
    for line in (l for l in text.splitlines() if l.strip()):
        obj = json.loads(line)
        if "message" in obj and "content" in obj["message"]:
            full_content += obj["message"]["content"]
        if obj.get("done"):
            break
    return full_content.strip()

def call_gemini(prompt: str, model :str, api_key: str = "", temperature: float = 0.0) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    gmodel = genai.GenerativeModel(model_name=model)
    resp = gmodel.generate_content(
        prompt,
        generation_config={"temperature": temperature},
    )
    return (resp.text or "").strip()

def generate_answer(query:str, retrieved_text: list[str], backend: str, model: str, api_key: str = "", ollama_url:str = "") ->str:
    prompt = build_prompt(query, retrieved_text)
    if backend == "Gemini (API)":
        return call_gemini(prompt, model, api_key)
    return call_ollama(prompt, model, ollama_url)

# sidebar --- setup 

st.sidebar.header("⚙️ Setup")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

backend = st.sidebar.selectbox(
    "Generation backend", ["Gemini (API)", "Ollama (local)"],
    help="Use Gemini for the deployed/live demo. Ollama only works when running locally "
         "with `ollama serve` up -- it will not work on a hosted Hugging Face Space.",
)

if backend == "Gemini (API)":
    gemini_api_key = st.sidebar.text_input(
        "Gemini API key", type="password",
        value= os.environ.get("Gemini_API_KEY", ""),
        help="Get a free key at https://aistudio.google.com/apikey. "
             "On Hugging Face Spaces, set this as a Space secret named GEMINI_API_KEY instead.",
    )
    gemini_model = st.sidebar.text_input("Gemini model", value="gemini-2.5-flash")
    ollama_url, ollama_model = "", ""

else:
    ollama_url = st.sidebar.text_input("Ollama URL", value="http://localhost:11434/api/chat")
    ollama_model = st.sidebar.text_input("ollama model", value="llama3.1:8b")
    gemini_api_key, gemini_model = "", ""

st.sidebar.divider()
top_k = st.sidebar.slider("Top k chunks retrieved ", 1, 8, 3)
use_rerank = st.sidebar.checkbox("Use cross encoder re-ranking", value=True)

st.sidebar.divider()
if st.sidebar.button("🪓Build / Rebuild Index", type="primary", disabled=uploaded_pdf is None):
    with st.spinner("processing PDF (loading, chunking, embedding)......"):
        pages = load_pdf_pages(uploaded_pdf.getvalue())
        chunks = chunk_pages(pages)
        index = build_index(chunks)
        st.session_state.index = index
        st.session_state.chunks = chunks
        st.session_state.pdf_name = uploaded_pdf.name
        st.session_state.chat_history = []
        st.session_state.debug_log = []

    st.sidebar.success(f"Indexed {len(chunks)} chunks from {len(pages)} pages.")
if "index" in st.session_state:
    st.sidebar.caption(f"📃 currently indexed : **{st.session_state.pdf_name}**"
                        f"{len(st.session_state.chunks)} chunks")
        

# Main area (chat + RAGAS Evaluation tabs)

st.title("📚 RAG chatbot")

tab_chat, tab_eval = st.tabs(["chat", "📊 RAGAS Evaluation"])

with tab_chat:
    if "index" not in st.session_state:
        st.info("upload a pdf clich\k **build / rebuild index in the sidebar tp get started.")
    else:
        for msg in st.session_state.get("chat_history", []):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander(f"{len(msg['sources'])} sources retrieved"):
                        for s in msg["sources"]:
                            score = s.get("re-rank_score", s["score"])
                            st.markdown(f"**pages {s['page_start']}{s['page_end']}** (score={score:.3f})")
                            st.caption(s["text"][:300] + "...")
        
        query = st.chat_input("Ask a question about the document")
        if query:
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):
                answer= ""
                with st.spinner("retrieveing relevant chaunks ......"):
                    retrieved = retrieve(
                        query, st.session_state.index, st.session_state.chunks,k= top_k, use_re_rank=use_rerank,
                    )
                retrieved_text = [r["text"] for r in retrieved]
                with st.spinner("Generatign answer ....."):
                    try:
                        answer = generate_answer(
                        query, retrieved_text, backend,
                        model=(gemini_model if backend == "Gemini (API)" else ollama_model),
                        api_key=gemini_api_key, ollama_url=ollama_url,
                    )
                    except Exception as e:
                        answer = (f"⚠️ Generation error: {e}\n\n"
                                  "If using Ollama, confirm `ollama serve` is running locally. "
                                  "If using Gemini, confirm your API key is set correctly.")

            st.write(answer)
            if retrieved:
                with st.expander(f" {len(retrieved)} sources retrieved "):
                    for s in retrieved:
                        score = s.get("re-rank_score", s["score"])
                        st.markdown(f"**pages {s['page_start']}-{s['page_end']}** (score={score:.3f})")
                        st.caption(s["text"][:300] + "...........")

            st.session_state.chat_history.append({
            "role": "assistant", "content": answer, "sources": retrieved
            })
            st.session_state.debug_log.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "query": query,
            "retrieved_chunks": [
                {"pages": f"{r['page_start']}-{r['page_end']}", 
                 "score": r.get("re-rank_score", r["score"]), "preview": r["text"][:200]}
                 for r in retrieved

                ],
                "answer": answer
            })                    

#----------------------RAGAS Evaluation ----------------------------------
with tab_eval:
    st.subheader("RAGAS Evaluation")
    if "index" not in st.session_state:
        st.info("Index  PDF in the chat tab first.")
    else :
        st.write(
            "Upload a CSV with a `question` column (and optional `reference` column for "
            "reference-based metrics), or paste questions below -- **up to 20 questions**. "
            "Faithfulness and response relevancy run on every question; context recall and "
            "factual correctness only run on questions that have a `reference` answer."
        )

        template_df = pd.DataFrame({
            "question": ["What are the four functions of the AI RMF Core?", "..."],
            "reference": ["GOVERN, MAP, MEASURE, and MANAGE.", "..."],
        })

        st.download_button(
            "download CSV template", template_df.to_csv(index=False),
            file_name="ragas_test_questions_template.csv", mime="text/csv",
        )

        eval_csv = st.file_uploader("Upload test questions csv", type=["csv"], key="eval_csv")
        manual_qs = st.text_area(

             "...or paste one question per line (no references, reference-free metrics only)",
            height=120
        )
        run_eval = st.button("Run RAGAS evaluation", type="primary")
        if run_eval:
            eval_items = []
            if eval_csv is not None:
                df_in = pd.read_csv(eval_csv)
                for _, row in df_in.head(20).iterrows():
                    eval_items.append({
                        "question": str(row["question"]),
                        "reference": str(row["reference"]) if "reference" in df_in.columns and pd.notna(row.get("reference")) else None,
                    })
            elif manual_qs.strip():
                lines = [l.strip() for l in manual_qs.splitlines() if l.strip()][:20]
                eval_items = [{"question": q, "reference": None} for q in lines]

            if not eval_items:
                st.warning("provide at least one question via csv upload or the text box")
            elif backend == "Gemini (API)" and not gemini_api_key:
                st.warning("Enter a Gemini API KEY in the sidebar first --- it's also used at the RAGAS judge model")
            else :
                from ragas import EvaluationDataset, evaluate
                from ragas.embeddings import LangchainEmbeddingsWrapper
                from ragas.llms import LangchainLLMWrapper
                from ragas.metrics import Faithfulness, FactualCorrectness, LLMContextRecall, ResponseRelevancy

                progress = st.progress(0.0, text="Running retrieveal + generation ...")
                rows = []
                for i, item in enumerate(eval_items):
                    q = item["question"]
                    retrieved = retrieve(q, st.session_state.index, st.session_state.chunks, k=top_k, use_re_rank=use_rerank)
                    retrieved_text = [r["text"] for r in retrieved]
                    try :
                            answer = generate_answer(
                                q, retrieved_text, backend, model= (gemini_model if backend == "Gemini (API)" else ollama_model),
                                api_key=gemini_api_key, ollama_url=ollama_url,
                            )
                    except Exception as e:
                            answer = f"[generation error: {e}]"
                    rows.append({"user_input": q, "retrieved_contexts": retrieved_text,
                                        "response": answer, "reference": item["reference"]})
                    progress.progress((i + 1) / len(eval_items),
                                       text=f"Running retrieval + generation... ({i+1}/{len(eval_items)})")
                    time.sleep(0.3)
                progress.empty()

                with st.spinner("Scoring with RAGAS"):
                    if backend == "Gemini (API)":
                            from langchain_google_genai import ChatGoogleGenerativeAI
                            judge_llm = LangchainLLMWrapper(
                                ChatGoogleGenerativeAI(model=gemini_model, google_api_key=gemini_api_key, temperature=0.0)
                            )
                    else :
                            from langchain_ollama import ChatOllama
                            judge_llm = LangchainLLMWrapper(ChatOllama(model=ollama_model, temperature=0.0))

                    from langchain_huggingface import HuggingFaceEmbeddings
                    judge_embeddings = LangchainEmbeddingsWrapper(
                            HuggingFaceEmbeddings(model_name=embedding_model)
                        )

                    ref_free_rows = [{**r, "reference": r["reference"] or ""} for r in rows]
                    ds_free = EvaluationDataset.from_list(ref_free_rows)
                    result_free = evaluate(dataset=ds_free, metrics=[Faithfulness(), ResponseRelevancy()],
                                               llm=judge_llm, embeddings=judge_embeddings)
                    free_df = result_free.to_pandas()

                    ref_rows = [r for r in rows if r["reference"]]
                    if ref_rows:
                            ds_ref = EvaluationDataset.from_list(ref_rows)
                            result_ref = evaluate(dataset=ds_ref, metrics=[LLMContextRecall(), FactualCorrectness()],
                                                  llm=judge_llm, embeddings=judge_embeddings)
                            ref_df = result_ref.to_pandas()
                    else :
                            ref_df = None
                st.session_state.ragas_free_df = free_df
                st.session_state.ragas_ref_df = ref_df

        if st.session_state.get("ragas_free_df") is not None:
            free_df= st.session_state.ragas_free_df
            st.markdown("### Aggregate scores")
            score_cols = st.columns(4)
            score_cols[0].metric("Faithfulness (avg)", f" {free_df['faithfulness'].mean():.2f}")
            score_cols[1].metric("Responce Relevancy (avg)", f"{free_df['answer_relevancy'].mean():.2f}" 
                                     if "answer_relevancy" in free_df.columns 
                                     else f"{free_df.get('responce_relevancy', pd.Series([float('nan')])).mean():.2f}")
            ref_df = st.session_state.get("ragas_ref_df")
            if ref_df is not None :
                    score_cols[2].metric("Context Recall (avg)", f"{free_df['context_recall'].mean():.2f}"
                                         if "context_recall" in ref_df.columns else "n/a")
                    score_cols[3].metric("Factual Correctness (avg)", f"{ref_df['factual_correctness'].mean():.2f}"
                                      if "factual_correctness" in ref_df.columns else "n/a")
                
            else :
                    score_cols[2].metric("context recall (avg)", "no reference")
                    score_cols[3].metric("Factual correctness (avg)", "no reference")

            st.markdown("### Per-question scores (reference-free)")
            st.dataframe(free_df, use_container_width=True)
            if ref_df is not None:
                    st.markdown("### Per-question scores (reference-based)")
                    st.dataframe(ref_df, use_container_width=True)
