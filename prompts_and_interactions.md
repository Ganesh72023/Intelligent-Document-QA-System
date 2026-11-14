# Conversation History & Technical Interactions: The Relay

**Candidate:** ML Engineering Intern Candidate  
**Project:** The Relay — Production-Grade RAG Engineering Assistant Under Strict API Limits  
**Model & Platform:** Groq LLaMA-3.1-8B-Instant / Google Antigravity Agent  

---

## 1. The Three Most Important Prompts and Why

### Prompt 1: Multi-Step Pipeline Design with Bounded Evidence Expansion
> *"Build an asynchronous 7-step RAG pipeline where retrieval failure triggers an autonomous bounded retry loop that expands the search window before deciding to answer or refuse."*

- **Why it mattered:** A naive single-turn RAG pipeline either answers when context is poor or blindly hallucinates. This prompt forced the architecture to treat retrieval as an active, verified loop with explicit validation and refusal safeguards.

### Prompt 2: Empirical Calibration of the Stand-in Simulator
> *"Do not invent arbitrary latency or rate limit numbers. Build the FastAPI stand-in model server to simulate Groq's free-tier behavior calibrated directly from real response headers (RPM=30, TPM=14,400, TPD=500,000, SSE streaming, and load-dependent latency degradation)."*

- **Why it mattered:** Simulated benchmarks are worthless if they model ideal systems. This prompt ensured the stand-in replicated real-world backpressure, SSE chunking delays, and concurrency bottlenecks.

### Prompt 3: Strict Ground-Truth Evaluation and Reproducibility Mandate
> *"Every single figure in the report tables must be directly computed from results.csv. Generate 10 Matplotlib graphs and evaluate 17 questions across Attention, Moshi, and Dynamic Chunking."*

- **Why it mattered:** It eliminated synthetic reporting. Every latency percentile (p50, p95, p99), token count, and outcome percentage in the final write-up matches the underlying dataset.

---

## 2. One Thing the AI Got Wrong and How It Was Caught

### The Issue:
During initial metrics implementation, the assistant suggested hardcoding token pricing at `$0.0005 per 1k tokens` to populate the `Cost per 1,000 answers` column.

### How It Was Caught:
Reviewing the assignment brief ground rules:
> *"Use your own free account. Do not invent pricing. If no monetary charge occurred, say so. If a paid cost cannot be measured or reliably established, state that it was not measured rather than inventing a price."*

### The Resolution:
I caught the violation in code review and required the metrics engine to record `$0.00 (Free)` for all free-tier executions and `not measured` where appropriate, ensuring complete academic honesty.

---

## 3. One Decision Where I Overruled the AI

### The AI Suggestion:
The AI proposed installing an external cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for neural reranking in Configuration C.

### My Overruling:
I overruled this decision. An external neural cross-encoder requires multi-gigabyte PyTorch GPU dependencies, increases RAM footprint past standard test environments, and adds 400ms of latency per query. Instead, I designed a deterministic keyword-boosted reranking algorithm that operates directly on normalized FAISS inner-product scores. This preserves CPU lightweight portability while providing measurable ranking improvements.

---

## 4. What Work and Decisions Are Distinctly Mine

1. **Decoupling Concurrency from Provider Invocations:** The in-memory token bucket rate limiter and async FIFO queue that buffers request bursts before they ever reach the network.
2. **Strict Ground-Truth Test Suite:** Constructing the 17 comprehensive test questions spanning all three arXiv papers (Attention Is All You Need, Moshi, and Dynamic Chunking) plus cross-paper synthesis and deliberate out-of-scope queries.
3. **Stand-in Load-Dependent Degradation Curve:** The mathematical formulation scaling latency by `(1 + (active_requests - 1) * 0.4 / max_concurrent)` to replicate real provider thread pool saturation.
4. **Transparent Failure Analysis:** The deep dive into slow request `sudden_burst-027`, isolating 1,900ms of pre-token queuing/backoff delay from 1,250ms of execution time.

---

## 5. Chronological Interaction Log

### Phase 1: Setup & Data Ingestion
- User provided master assignment brief.
- Assistant verified Python 3.10 environment.
- Assistant initiated download of exact papers:
  - `arXiv:1706.03762v7` (Attention Is All You Need) -> 2.21 MB
  - `arXiv:2410.00037v1` (Moshi) -> 4.69 MB
  - `arXiv:2507.07955v2` (Dynamic Chunking) -> 1.38 MB

### Phase 2: RAG Pipeline & Rate Limiting Engine
- Created `src/document_loader.py` with PyMuPDF section parsing.
- Created `src/chunking.py` supporting variable chunk sizes and word overlaps.
- Created `src/embeddings.py` using `all-MiniLM-L6-v2`.
- Created `src/retrieval.py` supporting FAISS vector indexing.
- Created `src/rate_limiter.py` tracking rolling 60s windows for RPM/TPM and daily budgets.
- Created `src/retry_manager.py` with exponential backoff and jitter.

### Phase 3: Simulator, Evaluations & Load Testing
- Implemented `simulator/standin_server.py` with FastAPI.
- Implemented `simulator/load_generator.py` covering all 5 load scenarios.
- Implemented `evaluation/evaluate_quality.py` for automated grading.
- Implemented `evaluation/generate_graphs.py` for 10 Matplotlib figures.
- Executed unit tests and verified reproducibility.
