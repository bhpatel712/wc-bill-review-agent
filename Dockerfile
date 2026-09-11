# WC Bill Review & Code Anomaly Detection Agent -- Week 6 container.
#
# Bakes in everything the API needs to run WITHOUT re-running the Weeks
# 1-4 pipeline: the trained model (models/*.pkl) and the pre-built RAG
# vector store (agent/chroma_db), plus the reference/corpus data the
# service reads at request time (data-generation/reference/*.csv,
# data-generation/output/synthetic_bills.jsonl).
#
# Deliberately does NOT bake in agent/.env (see .dockerignore) -- that
# file holds real Azure OpenAI credentials. Supply it at `docker run`
# time instead, so it only ever exists as environment variables inside
# a running container, never as a layer in the image itself.
#
# Build (from the project root):
#     docker build -t wc-bill-review-agent .
#
# Run, with RAG explanations available (mounts your real agent/.env as
# environment variables only -- the file itself is never copied in):
#     docker run --rm -p 8000:8000 --env-file agent/.env wc-bill-review-agent
#
# Run without agent/.env (fine -- /predict and /review still work; only
# ?generate_explanations=true will report "not configured"):
#     docker run --rm -p 8000:8000 wc-bill-review-agent
#
# Then open http://127.0.0.1:8000

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so `docker build` can reuse this layer
# across rebuilds that only touch source code, not requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Whole directories, not individual files -- agent/, api/,
# data-generation/, and models/ all read each other via relative paths
# (e.g. api/code_lookup.py reads agent/retrieve.py and
# data-generation/generate_bills.py; api/inference.py reads
# models/features.py and data-generation/output/synthetic_bills.jsonl;
# agent/retrieve.py reads agent/chroma_db and data-generation/reference).
# Cherry-picking individual files here is exactly how you end up
# silently missing one that a relative import needs -- see
# .dockerignore for what's deliberately excluded instead.
COPY agent/ ./agent/
COPY api/ ./api/
COPY data-generation/ ./data-generation/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
