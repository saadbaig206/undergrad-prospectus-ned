# RAG Improvement & Scalability Roadmap

This document outlines the advanced architectural improvements that can be introduced to **Prospectus AI** to make it faster, more accurate, and production-ready.

---

## 1. Advanced Retrieval Enhancements

### A. Hybrid Search (Dense + Sparse Retrieval)
- **Problem**: Vector search sometimes misses specific terms, alphanumeric course codes (e.g., `"MT-100"`), or exact names.
- **Solution**: Combine dense vector search with sparse TF-IDF or BM25 retrieval.
- **Implementation**:
  - Store sparse vectors in Pinecone alongside dense embeddings.
  - Calculate dense and sparse retrieval ranks.
  - Merge them using **Reciprocal Rank Fusion (RRF)**:
    $$RRF\_Score(d) = \sum_{m \in M} \frac{1}{60 + r_m(d)}$$

### B. Cross-Encoder Reranking
- **Problem**: The Bi-Encoder used for embedding query-to-document matching focuses on conceptual similarity but can include noisy/irrelevant chunks.
- **Solution**: Retrieve the top 20 candidate chunks, then pass them through a Cross-Encoder (Reranker) to evaluate the exact query-chunk interaction.
- **Implementation**:
  - Run a lightweight reranking model (e.g., `BAAI/bge-reranker-base`) or use an API (e.g., Cohere Rerank) to filter the 20 candidates down to the top 4 most relevant chunks before feeding them to the LLM.

### C. Query Translation & HyDE (Hypothetical Document Embeddings)
- **Problem**: Raw user queries are often brief or poorly phrased, leading to sub-optimal vector matches.
- **Solution**: Use the LLM to generate a hypothetical ideal answer first, embed that answer, and use the generated embedding to query the vector database.
- **Benefits**: Focuses the search on the matching answers rather than matching the query structure.

---

## 2. Infrastructure & Caching Upgrades

### A. Persistent Semantic Cache (Redis / pgvector)
- **Problem**: The in-memory cache is ephemeral. When deployed on serverless hosting (Vercel), the cache is wiped whenever the serverless container sleeps or restarts.
- **Solution**: Move the exact and semantic cache to a remote database.
- **Implementation**:
  - Use **Redis** or a PostgreSQL database with **pgvector** to store `[query, embedding_vector, cached_response]`.
  - Perform HNSW index searches on Redis/pgvector for $\ge 0.88$ cosine similarity.

### B. Dynamic Ingestion Pipelines (Chunking Optimization)
- **Problem**: Prospectus sections vary in layout (e.g., simple prose vs. curriculum grids vs. fee spreadsheets).
- **Solution**: Apply **Parent-Child Chunking** (or hierarchical chunking).
- **Implementation**:
  - Split documents into large parent chunks (e.g., 2000 chars for context preservation) and link them to multiple smaller child chunks (e.g., 300 chars for precise vector matching). When a child matches, feed the parent chunk to the LLM.

---

## 3. Data Structuring & Quantitative Querying

### A. Tabular Data Routing (Text-to-SQL / Pandas)
- **Problem**: Vector databases are notoriously bad at quantitative reasoning over tabular data (e.g., *"List all departments with self-finance fees under 100,000"*).
- **Solution**: Extract fee and seat distribution matrices into database tables.
- **Implementation**:
  - Use LLM routing to detect if a query is structural or math-heavy.
  - If yes, route the query to a **Text-to-SQL** generator or run structured Pandas queries on clean JSON files instead of vector matching.

---

## 4. Guardrails & Compliance

### A. LLM Response Guardrails
- **Problem**: Prevent the bot from answering inappropriate questions, leaking prompts, or halluncinating out-of-scope topics.
- **Solution**: Introduce **NeMo Guardrails** or a dual-LLM guardrail check.
- **Implementation**:
  - A fast, tiny model parses the generated response to check if it matches the safe-response criteria before returning the tokens to the client.
