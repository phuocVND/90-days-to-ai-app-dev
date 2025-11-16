# RAG Backend Codebase Analysis

**Last Updated:** 2025-11-16
**Analyzed By:** Senior Python Engineer with RAG Expertise
**Codebase:** MUTCD RAG Q&A System

---

## Executive Summary

This is a **proof-of-concept RAG (Retrieval Augmented Generation) system** designed to answer questions about the US Manual on Uniform Traffic Control Devices (MUTCD). The implementation is clean and well-structured for learning purposes, but requires significant enhancements for production deployment.

**Key Stats:**
- **Core Code:** 156 lines of Python
- **Data Size:** 2 PDFs (51 MB total)
- **Embedding Model:** all-MiniLM-L6-v2 (384 dimensions)
- **LLM:** BLOOM-560M (lightweight generative model)
- **Vector Storage:** In-memory (no persistence)
- **Query Latency:** 5-30+ seconds (CPU-bound LLM inference)

---

## 1. Architecture Overview

### 1.1 Layered Architecture

```
┌─────────────────────────────────────────┐
│        HTTP Layer (FastAPI)             │
│        - CORS Middleware                │
│        - Request/Response Validation    │
├─────────────────────────────────────────┤
│     API Routes (ask_router.py)          │
│        - POST /api/ask                  │
├─────────────────────────────────────────┤
│     Service Layer (service.py)          │
│        - get_prompt()                   │
│        - generate_answer()              │
├─────────────────────────────────────────┤
│     Model Layer (PDFModel)              │
│        - PDF Reading                    │
│        - Text Chunking                  │
│        - Embedding Generation           │
│        - Retrieval                      │
│        - Prompt Construction            │
│        - LLM Generation                 │
└─────────────────────────────────────────┘
```

### 1.2 Directory Structure

```
rag_backend/
├── app/
│   ├── __init__.py              # FastAPI app factory
│   ├── main.py                  # App initialization
│   ├── api/
│   │   └── routes/
│   │       └── ask_router.py    # Question answering endpoint
│   ├── core/
│   │   └── config.py            # Middleware configuration
│   ├── models/
│   │   └── model.py             # Core RAG implementation (99 LOC)
│   ├── schemas/
│   │   └── schema.py            # Pydantic models
│   └── services/
│       └── service.py           # Service layer (singleton)
├── data/
│   ├── mutcd11thedition.pdf     # Main MUTCD document (23 MB)
│   └── mutcd11theditionhl.pdf   # Highlighted version (28 MB)
├── run_server.py                # Server entry point
├── index.html                   # Web UI
├── api_test.rest                # API test examples
└── requirements.txt             # Dependencies
```

---

## 2. Technology Stack

### 2.1 Core Dependencies

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Web Framework** | FastAPI | Modern async API framework |
| **Server** | Uvicorn | ASGI server |
| **Validation** | Pydantic | Request/response validation |
| **Embeddings** | sentence-transformers | Text embeddings (all-MiniLM-L6-v2) |
| **LLM** | transformers + torch | LLM inference (BLOOM-560M) |
| **PDF Processing** | PyMuPDF (fitz) | PDF text extraction |
| **ML Utilities** | scikit-learn, numpy | Similarity calculations |

### 2.2 Model Details

**Embedding Model: all-MiniLM-L6-v2**
- Type: Sentence embedding model
- Dimensions: 384
- Size: ~23 MB
- Speed: Fast on CPU
- Purpose: Convert text to semantic vectors

**Language Model: bigscience/bloomz-560m**
- Type: Generative LLM
- Parameters: 560 million
- Size: ~1.1 GB
- Device: CPU (device=-1)
- Purpose: Generate natural language answers

---

## 3. RAG Pipeline Deep Dive

### 3.1 Document Ingestion

**Location:** [app/models/model.py:33-46](../app/models/model.py#L33-L46)

**Process:**
1. Scans `data/` folder for all PDF files
2. Uses PyMuPDF to extract text page-by-page
3. Warns about empty pages
4. Concatenates all text from all PDFs

**Code:**
```python
def _read_all_pdfs(self):
    """Reads and concatenates text from all PDFs in data folder"""
    all_text = []
    pdf_files = [f for f in os.listdir(self.data_dir) if f.endswith('.pdf')]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(self.data_dir, pdf_file)
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text = page.get_text()
                if text.strip():
                    all_text.append(text)
                else:
                    print(f"Warning: Page {page.number} in {pdf_file} has no text")

    return "\n".join(all_text)
```

### 3.2 Text Chunking Strategy

**Location:** [app/models/model.py:48-67](../app/models/model.py#L48-L67)

**Parameters:**
- `max_sentences = 1`: Each chunk is exactly one sentence
- `max_words = 1024`: Maximum 1024 words per chunk

**Chunking Algorithm:**

1. **Text Cleaning:**
   - Normalize whitespace (tabs → spaces, multiple spaces → single)
   - Remove non-ASCII characters
   - Strip leading/trailing whitespace

2. **Sentence Splitting:**
   - Regex: `(?<=[.!])\s+(?=[A-Z0-9])`
   - Splits on period/exclamation followed by capital letter or digit
   - Preserves sentence boundaries

3. **Chunk Assembly:**
   - Accumulates sentences until reaching `max_words`
   - Creates new chunk when limit exceeded
   - Respects `max_sentences` parameter

**Issues:**
- `max_sentences=1` with `max_words=1024` is contradictory (one sentence rarely has 1024 words)
- May create very large chunks for run-on sentences
- Regex splitting may miss some sentence boundaries

### 3.3 Embedding Generation

**Location:** [app/models/model.py:70-74](../app/models/model.py#L70-L74)

**Process:**
```python
def _prepare_data(self):
    text = self._read_all_pdfs()
    chunks = self._split_into_chunks(text, self.max_sentences, self.max_words)
    embeddings = self.embed_model.encode(chunks, show_progress_bar=True)
    return chunks, embeddings
```

**Characteristics:**
- Runs on initialization (blocking)
- Shows progress bar during encoding
- Stores all embeddings in memory
- No persistence across restarts

**Storage:**
- Chunks: List of strings
- Embeddings: numpy array of shape (n_chunks, 384)

### 3.4 Retrieval Mechanism

**Location:** [app/models/model.py:76-80](../app/models/model.py#L76-L80)

**Algorithm:**
```python
def get_top_chunks(self, question, top_k=2):
    # 1. Encode question
    q_emb = self.embed_model.encode([question])[0]

    # 2. Calculate cosine similarity with all chunks
    sims = [
        np.dot(q_emb, c_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(c_emb))
        for c_emb in self.chunk_embeddings
    ]

    # 3. Get top K indices
    top_idx = np.argsort(sims)[::-1][:top_k]

    # 4. Return corresponding chunks
    return [self.chunks[i] for i in top_idx]
```

**Similarity Metric:** Cosine Similarity
- Formula: `cos(θ) = (A · B) / (||A|| × ||B||)`
- Range: -1 to 1 (higher = more similar)
- Computed using numpy operations

**Performance:**
- Time Complexity: O(n) per query (linear scan)
- Space Complexity: O(n) for similarity scores
- No indexing or optimization (FAISS, HNSW, etc.)

### 3.5 Prompt Construction

**Location:** [app/models/model.py:82-89](../app/models/model.py#L82-L89)

**Template:**
```python
def build_prompt(self, question, top_k):
    context = "\n".join(self.get_top_chunks(question, top_k))

    prompt = (
        f"Question:\n{question}\n\n"
        "Please answer the following question based strictly on the "
        "reference document provided below.\n\n"
        f"Reference document:\n{context}\n"
    )

    return prompt
```

**Prompt Structure:**
1. Question section
2. Instruction to answer based on reference
3. Context section with retrieved chunks

**Limitations:**
- No system prompt
- No few-shot examples
- Generic instruction (not domain-specific)
- No citation or source attribution

### 3.6 LLM Generation

**Location:** [app/models/model.py:91-98](../app/models/model.py#L91-L98)

**Configuration:**
```python
def answer_question(self, prompt):
    return self.qa_pipeline(
        prompt,
        max_new_tokens=512,         # Maximum output length
        temperature=0.3,            # Low temp = deterministic
        top_p=0.9,                  # Nucleus sampling
        do_sample=True              # Enable sampling
    )[0]['generated_text']
```

**Parameters Explained:**
- **max_new_tokens=512**: Limits response length (prevents runaway generation)
- **temperature=0.3**: Low value favors high-probability tokens (more factual, less creative)
- **top_p=0.9**: Nucleus sampling keeps top 90% probability mass (balances diversity and quality)
- **do_sample=True**: Uses stochastic sampling instead of greedy decoding

**Issues:**
- No GPU support (CPU-only inference is slow)
- Synchronous execution (blocks event loop)
- No timeout handling
- Returns full generated text (includes prompt repetition)

---

## 4. API Interface

### 4.1 Single Endpoint: POST /api/ask

**Location:** [app/api/routes/ask_router.py](../app/api/routes/ask_router.py)

**Request Schema:**
```python
class QuestionRequest(BaseModel):
    question: str      # User's question
    top_k: int = 3     # Number of chunks to retrieve (default: 3)
```

**Response Schema:**
```python
class AnswerResponse(BaseModel):
    question: str      # Echo of input question
    prompt: str        # Full prompt sent to LLM
    answer: str        # Generated answer
```

**Example Request:**
```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the standard color for a stop sign?",
    "top_k": 2
  }'
```

**Example Response:**
```json
{
  "question": "What is the standard color for a stop sign?",
  "prompt": "Question:\nWhat is the standard color for a stop sign?\n\nPlease answer...",
  "answer": "According to the reference document, a stop sign must be red with white lettering."
}
```

### 4.2 CORS Configuration

**Location:** [app/core/config.py](../app/core/config.py)

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # ⚠️ Allow all origins
    allow_credentials=True,
    allow_methods=["*"],           # ⚠️ Allow all methods
    allow_headers=["*"],           # ⚠️ Allow all headers
)
```

**Security Risk:** Wide-open CORS suitable for development only

---

## 5. Service Layer

**Location:** [app/services/service.py](../app/services/service.py)

```python
from app.models.model import PDFModel

# Singleton pattern - initialized once on import
pdf_model = PDFModel()

def get_prompt(question: str, top_k: int = 2) -> str:
    """Build prompt from question and retrieved context"""
    return pdf_model.build_prompt(question, top_k)

def generate_answer(prompt) -> str:
    """Generate answer from prompt using LLM"""
    return pdf_model.answer_question(prompt)
```

**Design:**
- Module-level singleton
- Thin wrapper around PDFModel
- No error handling
- No logging

**Implications:**
- PDFModel initializes when module imports
- Blocks server startup for initialization time
- Shared state across all requests
- Difficult to test (hard to mock)

---

## 6. Performance Profile

### 6.1 Initialization Metrics

**Estimated Timeline:**
1. PDF Reading: 10-30 seconds (depends on file size)
2. Text Chunking: 1-5 seconds
3. Embedding Generation: 30-120 seconds (depends on chunk count)
4. **Total: 40-155 seconds** (blocks server startup)

**Memory Usage:**
- PDFs: 51 MB on disk
- Extracted text: ~50-100 MB in memory
- Chunks: ~20-50 MB (depends on count)
- Embeddings: chunks × 384 × 4 bytes
  - Example: 5000 chunks = ~7.5 MB
- Model weights: ~1.2 GB (embedding + LLM)
- **Total: ~1.3-1.5 GB**

### 6.2 Query Latency Breakdown

| Operation | Time | Optimization Potential |
|-----------|------|----------------------|
| Question Embedding | 50-200ms | Low (already fast) |
| Similarity Search | 10-100ms | High (use vector DB) |
| Prompt Building | <1ms | Low (trivial) |
| LLM Generation | 5-30 seconds | Very High (use GPU, async) |
| **Total** | **5-30+ seconds** | **High** |

**Bottleneck:** CPU-based LLM inference dominates latency

### 6.3 Scalability Limits

**Current Constraints:**
- In-memory storage limits document size
- Linear scan limits chunk count (practical limit: ~100k chunks)
- Synchronous LLM blocks concurrent requests
- Single process (no horizontal scaling)

**Breaking Points:**
- Memory: >50 GB RAM for millions of chunks
- CPU: Single request can saturate CPU for 30 seconds
- Throughput: ~2-12 requests/minute (depends on LLM speed)

---

## 7. Code Quality Assessment

### 7.1 Strengths

✅ **Clean Architecture:**
- Well-organized layered structure
- Clear separation of concerns (routes, services, models)
- Follows FastAPI best practices

✅ **Type Safety:**
- Pydantic schemas for validation
- Type hints in function signatures

✅ **Readable Code:**
- Clear naming conventions
- Logical function decomposition
- Self-documenting code structure

✅ **Modern Stack:**
- FastAPI for async support
- Sentence transformers for embeddings
- Hugging Face transformers for LLM

### 7.2 Critical Issues

❌ **No Error Handling:**
```python
# No try-except blocks anywhere
def generate_answer(prompt) -> str:
    return pdf_model.answer_question(prompt)  # What if this fails?
```

**Risks:**
- Server crashes on PDF read errors
- No graceful degradation
- No user-friendly error messages

❌ **No Logging:**
- No visibility into system behavior
- Cannot debug production issues
- No audit trail for requests

❌ **No Persistence:**
```python
# Embeddings lost on restart
embeddings = self.embed_model.encode(chunks, show_progress_bar=True)
```

**Impacts:**
- Must recompute embeddings on every restart
- Cannot update documents incrementally
- Cannot scale across multiple servers

❌ **Blocking Initialization:**
```python
# Runs on import, blocks server startup
pdf_model = PDFModel()
```

**Problems:**
- Server unavailable for 1-2 minutes during startup
- No health checks or readiness probes
- Cannot test without waiting for initialization

❌ **No Async LLM:**
```python
# Synchronous generation blocks event loop
def answer_question(self, prompt):
    return self.qa_pipeline(prompt, ...)  # Blocks for 5-30 seconds
```

**Consequences:**
- Single request monopolizes server
- Other requests must wait
- Poor throughput under load

### 7.3 Security Vulnerabilities

🔒 **CORS Wide Open:**
```python
allow_origins=["*"]  # Any website can call this API
```

🔒 **No Authentication:**
- No API keys
- No rate limiting
- Open to abuse

🔒 **No Input Sanitization:**
```python
# No validation of question length or content
question: str  # Could be 1 MB of text
```

🔒 **No HTTPS:**
- HTTP only (development setup)
- Credentials/data transmitted in cleartext

---

## 8. Testing Coverage

### 8.1 Current State

**Test Files:** None
**Coverage:** 0%

**Available Testing:**
- Manual API tests in `api_test.rest` (23 examples)
- No unit tests
- No integration tests
- No load tests

### 8.2 Gaps

**Missing Tests:**
1. **Unit Tests:**
   - PDF reading with various file formats
   - Text chunking edge cases (empty text, very long sentences)
   - Embedding generation
   - Similarity calculation accuracy
   - Prompt construction

2. **Integration Tests:**
   - End-to-end question answering
   - API request/response validation
   - Error handling scenarios

3. **Performance Tests:**
   - Initialization time
   - Query latency
   - Memory usage
   - Concurrent request handling

4. **Regression Tests:**
   - Answer quality for known questions
   - Model output consistency

---

## 9. Production Readiness Gaps

### 9.1 Critical (Blockers)

| Gap | Impact | Effort |
|-----|--------|--------|
| No vector database | Cannot scale | High |
| No error handling | System crashes | Medium |
| No logging | Cannot debug | Low |
| Synchronous LLM | Poor throughput | High |
| No persistence | Must re-init on restart | High |

### 9.2 Important (Required)

| Gap | Impact | Effort |
|-----|--------|--------|
| No authentication | Security risk | Medium |
| No rate limiting | Abuse potential | Low |
| No monitoring | No observability | Medium |
| No health checks | Deployment issues | Low |
| No configuration | Hard to deploy | Low |

### 9.3 Nice-to-Have (Optional)

| Gap | Impact | Effort |
|-----|--------|--------|
| No caching | Higher latency | Medium |
| No tests | Hard to maintain | High |
| No documentation | Poor DX | Low |
| No CI/CD | Manual deployments | Medium |
| No GPU support | Slower inference | Low |

---

## 10. Key File Reference

| File Path | Purpose | Lines | Key Components |
|-----------|---------|-------|----------------|
| [app/models/model.py](../app/models/model.py) | Core RAG logic | 99 | PDFModel, embedding, retrieval, LLM |
| [app/services/service.py](../app/services/service.py) | Service layer | 9 | Singleton, get_prompt, generate_answer |
| [app/api/routes/ask_router.py](../app/api/routes/ask_router.py) | API endpoint | 11 | POST /api/ask |
| [app/schemas/schema.py](../app/schemas/schema.py) | Data models | 10 | QuestionRequest, AnswerResponse |
| [app/core/config.py](../app/core/config.py) | Configuration | 7 | CORS middleware |
| [run_server.py](../run_server.py) | Entry point | 10 | Uvicorn server launch |
| [requirements.txt](../requirements.txt) | Dependencies | 12 | Package list |

---

## 11. Domain-Specific Notes

### 11.1 MUTCD Context

**Document Type:** US Manual on Uniform Traffic Control Devices
**Size:** 1000+ pages
**Content:** Traffic signs, signals, markings, regulations
**Complexity:** Technical, regulatory language

### 11.2 Question Examples

From [api_test.rest](../api_test.rest), the system is designed to answer questions like:

- "What is a traffic control device?"
- "What does a red traffic signal mean?"
- "When must drivers yield to pedestrians?"
- "What is the standard color for a stop sign?"
- "Are U-turns permitted at intersections?"

**Answer Quality Factors:**
- Accuracy depends on retrieval quality
- LLM may hallucinate if context is insufficient
- No citation mechanism to verify sources
- Temperature 0.3 helps reduce creativity

---

## 12. Conclusion

This RAG backend represents a **functional proof-of-concept** with clean architecture and reasonable technology choices. However, it requires significant enhancements for production use, particularly in error handling, persistence, scalability, and performance optimization.

**Best Use Cases:**
- Educational demonstrations
- Prototype development
- Small-scale testing
- Learning RAG concepts

**Not Recommended For:**
- Production deployments
- Multi-user systems
- Real-time applications
- Large-scale document collections

---

## Next Steps

See [03_EXPERT_RECOMMENDATIONS.md](./03_EXPERT_RECOMMENDATIONS.md) for detailed improvement suggestions from a senior RAG engineer.
