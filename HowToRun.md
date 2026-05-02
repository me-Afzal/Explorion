# Explorion — Setup & Run Guide

Explorion is an agentic AI research platform powered by LangGraph, Ollama (local LLMs), Tavily web search, Pinecone vector memory, and a Streamlit UI.

---

## Prerequisites

### 1. Ollama (required)

Ollama runs the AI models locally on your machine.

- Download: https://ollama.com/download
- Available for Windows, Mac, and Linux

Pull the required models after installing:

```bash
ollama pull qwen2.5:7b    # used for research + analyze agents (required)
```

Make sure Ollama is running:
```bash
ollama serve
```

### 2. Docker (required for Option A only)

Docker is needed if you want to run Explorion via the pre-built image.

- Download Docker Desktop: https://www.docker.com/products/docker-desktop
- Available for Windows and Mac
- Linux: https://docs.docker.com/engine/install

Verify Docker is installed:
```bash
docker --version
```

---

## Step 1 — Set Up Your .env File

Copy the example file and fill in your keys:

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder values. All services are **free** — no credit card required:

| Variable | Service | How to get it |
|---|---|---|
| `PINECONE_API_KEY` | [Pinecone](https://app.pinecone.io) | Sign up → API Keys → Create API Key |
| `TAVILY_API_KEY` | [Tavily](https://app.tavily.com) | Sign up → Dashboard → API Keys |
| `LANGSMITH_API_KEY` | [LangSmith](https://smith.langchain.com) | Sign up → Settings → API Keys |

> Your `.env` file is never included in the Docker image. It is injected at runtime only.

---

## Option A — Docker (Recommended)

No Python or dependency setup needed. Just Ollama + Docker.

### Pull the pre-built image

```bash
docker pull afzal23/explorionv2:latest
```

### Run with your .env file

```bash
docker run --env-file .env -p 8501:8501 afzal23/explorionv2:latest
```

Open your browser at **http://localhost:8501**

> **Linux users:** Add `--add-host=host.docker.internal:host-gateway` so the container can reach Ollama:
> ```bash
> docker run --env-file .env -p 8501:8501 --add-host=host.docker.internal:host-gateway afzal23/explorionv2:latest
> ```

### Or use docker-compose (easier)

Create a `docker-compose.yml` file in the same folder as your `.env`:

```yaml
services:
  explorion:
    image: afzal23/explorionv2:latest
    ports:
      - "8501:8501"
    env_file:
      - .env
    extra_hosts:
      - "host.docker.internal:host-gateway"
    restart: unless-stopped
```

Then run:

```bash
docker-compose up
```

Open your browser at **http://localhost:8501**

> The `extra_hosts` line is required on Linux so the container can reach Ollama on the host. On Windows and Mac Docker Desktop it works automatically without it.

---

## Option B — Run Locally (Clone & Streamlit)

### 1. Clone the repo

```bash
git clone https://github.com/afzal23/explorion.git
cd explorion
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your .env file

```bash
cp .env.example .env
# edit .env with your keys
```

### 5. Run

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

> The `.env` file is loaded automatically from the project root on startup.

---

## Customization

### Change the AI models

By default both agents use `qwen2.5:7b`. You can upgrade the **manager agent** to a larger models(qwen2.5:14b,gemma models) for better instruction following — this is optional but recommended if you have the hardware.

Edit `assistant/llm_init.py`:

```python
def get_manager_llm() -> ChatOpenAI:
    return _make_ollama_llm("qwen2.5:7b")    # optional: upgrade to qwen2.5:14b for better results

def get_agent_llm() -> ChatOpenAI:
    return _make_ollama_llm("qwen2.5:7b")    # swap model name here
```

Pull the model first, then update the name in `llm_init.py`:

```bash
# Recommended upgrade for manager agent (~9 GB, needs ~10 GB RAM)
ollama pull qwen2.5:14b

# Other options
ollama pull gemma2:9b
ollama pull qwen3.5:9b
```

### Rebuild the Docker image after customization

```bash
docker build -t explorion .
docker run --env-file .env -p 8501:8501 explorion
```

---

## How the Ollama URL is resolved

Explorion auto-detects the right Ollama URL — no manual configuration needed:

| Environment | URL used |
|---|---|
| Running in Docker | `http://host.docker.internal:11434/v1` (auto) |
| Running locally | `http://localhost:11434/v1` (auto) |
| Custom override | Set `OLLAMA_BASE_URL=...` in your `.env` |
