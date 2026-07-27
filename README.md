# GCP-GPT

# GCP Copilot

**Ask questions about your Google Cloud infrastructure in plain English.**

GCP Copilot connects to a GCP project, pulls a complete inventory of every deployed resource, indexes it into a vector database, and lets engineers query it conversationally — *"which buckets are publicly accessible?"*, *"what's running in us-east1?"*, *"generate Terraform to replicate this VPC setup."*

Instead of clicking through the Cloud Console or writing `gcloud` queries by hand, you point it at a project ID and start asking.

<!-- TODO: add a screenshot or 15-second GIF of the Streamlit app here. This is the single highest-value addition to this README. -->

---

## Why

Auditing an unfamiliar GCP project is slow. The information is spread across the Console, `gcloud`, and asset exports, and answering a question like *"what does this client's networking actually look like?"* means manually assembling context from a dozen places.

GCP Copilot collapses that into a single ingestion run and a chat box.

---

## How it works

```
GCP Project
    │
    │  Cloud Asset Inventory API
    ▼
Resource inventory (JSON)
    │
    │  staged to S3
    ▼
Ingestion pipeline  ──  runs on Modal (serverless)
    │                   • parses 10+ file formats
    │                   • chunks to ~4k characters
    │                   • embeds via OpenAI text-embedding-3-small
    ▼
Pinecone index (1536-dim)
    │
    │  semantic search, top-k retrieval
    ▼
GPT-4o-mini  ──  grounded answer + generated code
```

**Ingestion is not limited to GCP asset JSON.** The pipeline handles PDF, DOCX, PPTX, XLSX/XLS, CSV, TXT, JSON, MSG (Outlook email), ZIP archives, and PNG — images are described via GPT-4 Vision before embedding. This means architecture diagrams, client documentation, and email threads can be indexed alongside the live infrastructure inventory and queried together.

Each ingestion run is recorded in an S3-backed log mapping source file to Pinecone index, so the query layer always resolves to the most recent snapshot automatically.

---

## Components

| File | Role |
|---|---|
| `app3.py` | Main Streamlit app — inventory fetch, ingestion, and Q&A in one UI |
| `inventory.py` | Standalone inventory extraction and ingestion trigger |
| `ingestion_script.py` | Multi-format parsing, chunking, embedding, and Pinecone upsert (runs on Modal) |
| `GCP_GPT_Assistant.py` | Retrieval and answer generation over the indexed data |
| `login.py` | Authentication gate |

---

## Setup

**Requirements:** Python 3.10+, a GCP service account with Cloud Asset Inventory read access, and API keys for OpenAI, Pinecone, and AWS.

```bash
git clone https://github.com/19UroojKhan/GCP-GPT.git
cd GCP-GPT
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET=your-bucket-name
S3_PREFIX=copilot/
```

Run:

```bash
streamlit run app3.py
```

Then upload your GCP service account JSON, enter a project ID, and fetch the inventory. Once ingestion completes, the Q&A tab is live.

To run ingestion as a standalone serverless job:

```bash
modal run ingestion_script.py
```

---

## Service account permissions

The service account needs `roles/cloudasset.viewer` on the target project. This is read-only — GCP Copilot never mutates infrastructure.

---



<!-- TODO: pick one. MIT is the default for a portfolio project. -->
