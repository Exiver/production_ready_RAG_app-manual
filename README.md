---
title: RAG Chatbot
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.40.0"
app_file: app.py
pinned: false
---

# Production RAG Chatbot

Upload any PDF → advanced retrieval (FAISS + cross-encoder re-ranking) → chat with it → evaluate answer quality with RAGAS, all in one Streamlit app.

## Features
- **Ingest any PDF** via the sidebar uploader — no fixed dataset
- **Advanced retrieval**: naive top-k FAISS search, or cross-encoder re-ranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) toggle
- **Dual generation backend**: Gemini (API, works everywhere including hosted deployment) or Ollama (local only, needs `ollama serve` running on the same machine)
- **In-UI RAGAS evaluation**: upload up to 20 test questions (with optional reference answers) and get faithfulness / response relevancy / context recall / factual correctness scores rendered directly in the app
- **Source transparency**: every answer shows the exact retrieved chunks (page numbers + scores) it was grounded in

## Running locally

```bash
# in your ai-engineering conda env
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

### Backend options while running locally
- **Ollama** (fully offline): make sure `ollama serve` is running and you've pulled a model (`ollama pull qwen2.5:7b`). Select "Ollama (local)" in the sidebar.
- **Gemini** (API, free tier): get a key at https://aistudio.google.com/apikey, paste it into the sidebar field, or set it as an environment variable before launching: `set GEMINI_API_KEY=your_key_here` (Windows cmd) or `$env:GEMINI_API_KEY="your_key_here"` (PowerShell).

## Deploying to Hugging Face Spaces (free)

**Important:** use **Gemini**, not Ollama, for the deployed demo. Free HF Spaces don't give you a persistent background process to run an Ollama server, and you can't realistically download multi-GB local models on the free CPU tier — the whole point of Ollama (a local server process) doesn't translate to a stateless hosted container. Gemini's free API tier works fine there since it's just an outbound HTTPS call.

1. Create a new Space at https://huggingface.co/new-space
   - SDK: **Streamlit**
   - Hardware: free **CPU basic** is enough (embedding + cross-encoder models are both small)
2. Push these three files to the Space repo: `app.py`, `requirements.txt`, `README.md` (this file — the YAML frontmatter above is what tells the Space it's a Streamlit app)
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<your-space-name>
   cd <your-space-name>
   cp /path/to/app.py /path/to/requirements.txt /path/to/README.md .
   git add .
   git commit -m "Initial RAG chatbot"
   git push
   ```
3. In the Space's **Settings → Variables and secrets**, add a secret named `GEMINI_API_KEY` with your key. The app reads `os.environ.get("GEMINI_API_KEY", "")` as the sidebar field's default, so it'll be pre-filled — no code change needed.
4. Wait for the build to finish (first build installs `sentence-transformers`/`torch`, so it can take several minutes), then open your Space's public URL — that's your live demo link.
5. In the app itself, select **Gemini (API)** as the backend, upload a PDF, click **Build / Rebuild Index**, and you're live.

## Project structure
```
app.py             # the entire app (ingestion, retrieval, generation, chat UI, RAGAS tab)
requirements.txt   # pinned-loosely dependencies for HF Spaces
README.md          # this file (also the HF Spaces config via YAML frontmatter)
```

## Notes on the RAGAS tab
- Faithfulness and Response Relevancy run on every question you provide, no reference answer needed.
- Context Recall and Factual Correctness only run on questions where you've filled in a `reference` (ground-truth) answer — write these yourself after reading the actual PDF, don't invent them.
- The evaluator ("judge") LLM is the same model you picked for generation. Worth knowing as a limitation: using the same model as both generator and judge risks self-preference bias in the scores.
