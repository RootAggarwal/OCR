"""RAG knowledge base for the OCR pipeline (table lane = Postgres, prose lane = pgvector).

Everything here is best-effort and OFF by default: the OCR pipeline must keep working
whether or not the RAG stack (Postgres, embedding model, LLM) is configured. Enable
document ingestion with RAG_INGEST_ENABLED=1 once Postgres is reachable.
"""
