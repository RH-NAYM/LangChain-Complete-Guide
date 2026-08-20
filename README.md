# 🦜🔗 LangChain Complete Guide

A hands-on, notebook-driven walkthrough of LangChain — covering the theory of foundation models plus practical implementations of **tools**, **agents**, and **RAG (Retrieval-Augmented Generation)**.

---

## 📁 Repository Contents

| File | Type | Description |
|---|---|---|
| `agent-langchain.ipynb` | Notebook | Building a **ReAct agent** with Ollama + DuckDuckGo search, including streaming responses |
| `tools-in-langchain.ipynb` | Notebook | Built-in and custom **LangChain tools** — 3 different ways to define tools, plus toolkits |
| `rag-using-langchain.ipynb` | Notebook | A full **RAG pipeline** that answers questions from a YouTube video transcript |
| `demoTranscript.txt` | Data | Fallback transcript (Lex Fridman × Demis Hassabis interview) used when the YouTube Transcript API is unavailable |
| `requirements.txt` | Config | Python dependencies for all notebooks |
| `README.md` | Docs | This file |

---

## 🧠 Foundation Models — Conceptual Map

This guide is organized around two complementary perspectives on foundation models:

### 1. Builder's Perspective
```
Transformer Architecture
├── Types of Transformers
│   ├── Encoder-only   (e.g. BERT)
│   ├── Decoder-only   (e.g. GPT)
│   └── Encoder-Decoder (e.g. T5)
├── Pretraining
│   ├── Training Objectives
│   ├── Tokenization Strategies
│   ├── Training Strategies
│   └── Handling Challenges
├── Optimization
│   ├── Training Optimization
│   ├── Model Compression
│   └── Inference Optimization
├── Fine-Tuning
│   ├── Task-Specific Tuning (RLHF)
│   ├── Instruction Tuning (PEFT)
│   └── Continual Pretraining
├── Evaluation
└── Deployment
```

### 2. User's Perspective
```
Building Basic LLM Apps
├── Open / Closed-source LLMs
├── Using LLM APIs
├── LangChain
├── HuggingFace
└── Ollama

Prompt Engineering
RAG
Fine-Tuning
Agents
LLMOps
Miscellaneous
```

---

## 📓 Notebook Breakdown

### 1️⃣ `tools-in-langchain.ipynb` — Tools 101

Explores how LangChain tools work, from built-ins to fully custom implementations:

- **Built-in tools**
  - `DuckDuckGoSearchRun` — web search
  - `ShellTool` — run shell commands
- **Custom tools — 3 methods**
  1. `@tool` decorator (simplest)
  2. `StructuredTool.from_function()` with a Pydantic input schema
  3. Subclassing `BaseTool` for full control
- **Toolkits** — grouping related tools (`add`, `multiply`) into a reusable `MathToolKit` class

### 2️⃣ `agent-langchain.ipynb` — Building a ReAct Agent

Walks through assembling an agent end-to-end:

1. Load `DuckDuckGoSearchRun` as a tool and set up a LangSmith `Client`
2. Instantiate a local model via `ChatOllama` (`gemma4:e4b`)
3. Pull the community **`hwchase17/react`** prompt template from LangSmith
4. Strip templated placeholders and build the agent with `create_agent(model, tools, system_prompt)`
5. Query the agent (e.g. *"3 ways to travel to Cox's Bazar from Dhaka + budget for a 3D2N trip"*)
6. Inspect the full message trace, then **stream** the response two ways:
   - `stream_mode="values"` — full state at each step
   - `stream_mode="messages"` — token-by-token streaming, including tool/agent update events

### 3️⃣ `rag-using-langchain.ipynb` — RAG over a YouTube Video

Builds a Retrieval-Augmented Generation pipeline that lets you "chat" with a YouTube video:

1. **Transcript ingestion** — fetches the transcript via `YouTubeTranscriptApi`, with automatic fallback to `demoTranscript.txt` if transcripts are disabled or the API fails
2. **Chunking** — `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap)
3. **Embeddings + Vector Store** — `OllamaEmbeddings` (`embeddinggemma:300m`) indexed into **FAISS**
4. **Retrieval** — top-5 similarity search
5. **Chain composition** using LCEL (`RunnableParallel`, `RunnablePassthrough`, `RunnableLambda`):
   ```
   {context, question} → prompt → OllamaLLM → StrOutputParser
   ```
6. Runs a sample query: *"Summarize the video in 5 points"* and prints the chain graph + final answer

> 📼 The bundled `demoTranscript.txt` is a transcript of a Lex Fridman podcast interview with **Demis Hassabis** (CEO & co-founder of DeepMind), covering AlphaZero, AlphaFold, and AI research.

---

## ⚙️ Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed locally, with the following models pulled:
  ```bash
  ollama pull gemma4:e4b
  ollama pull embeddinggemma:300m
  ```

### Installation
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file for any hosted-API integrations you plan to use:
```env
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
LANGSMITH_API_KEY=your_key_here
LANGSMITH_TRACING=true
```

---

## 📦 Key Dependencies

| Category | Libraries |
|---|---|
| **LangChain Core** | `langchain`, `langchain-core`, `langchain-community`, `langchain-text-splitters`, `langchain-experimental` |
| **Model Providers** | `langchain-openai`, `langchain-anthropic`, `langchain-ollama`, `langchain-google-genai`, `langchain-huggingface` |
| **RAG / Vector Store** | `faiss-cpu`, `tiktoken`, `youtube-transcript-api` |
| **Agents / Search** | `duckduckgo-search`, `ddgs` |
| **ML Utilities** | `numpy`, `scikit-learn`, `transformers`, `huggingface-hub` |
| **Misc** | `python-dotenv`, `pydantic`, `IPython`, `grandalf` (for ASCII graph rendering) |

---

## ▶️ Running the Notebooks

```bash
jupyter notebook
```
Then open any of:
- `tools-in-langchain.ipynb`
- `agent-langchain.ipynb`
- `rag-using-langchain.ipynb`

> 💡 Make sure your local Ollama server is running (`ollama serve`) before executing the agent or RAG notebooks, since they depend on local model inference and embeddings.

---

## 🗺️ Suggested Learning Path

1. Start with **Foundation Models** concepts above for context
2. `tools-in-langchain.ipynb` → understand how tools are defined and structured
3. `agent-langchain.ipynb` → see tools combined with a ReAct agent loop
4. `rag-using-langchain.ipynb` → apply retrieval + generation to real unstructured data

---

## 📝 Notes
- The agent notebook uses `dangerously_pull_public_prompt=True` to pull the public `hwchase17/react` prompt from LangSmith — review any pulled prompts before production use.
- The RAG notebook gracefully degrades to a local transcript file if the YouTube Transcript API is rate-limited or the video has transcripts disabled — useful for offline demos.
