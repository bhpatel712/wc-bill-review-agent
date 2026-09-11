"""
build_index.py  (WK4.1)

Embeds the ~60 rule documents from rule_docs.py and persists them to a
local Chroma vector store (agent/chroma_db/). Uses FastEmbed
(BAAI/bge-small-en-v1.5) for embeddings -- a small, free, local model, not
a paid API. That's a deliberate choice, not just a cost shortcut: this
reference corpus barely ever changes (it only grows when a new NCCI/MUE
update is folded in, per DRIFT_MONITORING.md's WK3 discussion of code-set
churn), so there's no ongoing reason to pay a per-call embeddings API for
something re-embedded rarely, in bulk, entirely offline. Azure OpenAI is
reserved for WK4.3's generation step, where a stronger hosted model
actually earns its cost.

First run downloads the embedding model (a few hundred MB, one-time,
cached locally afterward) -- needs internet access for that one time only.

Run:
    python3 build_index.py
"""
import os
import shutil

from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

from rule_docs import build_all_docs

HERE = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(HERE, "chroma_db")
COLLECTION_NAME = "wc_bill_review_rules"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def build_index():
    docs = build_all_docs()
    # deterministic ids so rerunning this script re-embeds cleanly instead
    # of accumulating duplicate documents on every run
    ids = []
    for d in docs:
        m = d.metadata
        if m["rule_type"] == "unbundling":
            ids.append(f"ncci_{m['code1']}_{m['code2']}")
        elif m["rule_type"] == "fragmented_billing":
            ids.append(f"mue_{m['code']}")
        else:
            ids.append(f"{m['rule_type']}_{m['topic']}")

    if os.path.isdir(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vectorstore.add_documents(documents=docs, ids=ids)

    print(f"indexed {len(docs)} rule documents -> {PERSIST_DIR}")
    print(f"embedding model: {EMBEDDING_MODEL}")
    print(f"collection: {COLLECTION_NAME}")
    return vectorstore


if __name__ == "__main__":
    build_index()
