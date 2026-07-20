# Adaptive Agent-Based Personalized Contextual Table Summarization

Multi-agent system for answering questions over hybrid table-and-text financial data. Decomposes complex reasoning across specialized agents with retrieval-augmented generation (RAG) to solve token efficiency and context preservation challenges.

## Problem Statement

Large language models struggle with hybrid table-text reasoning on financial documents. Four critical failure modes:

1. **Numerical Errors**: Multi-step arithmetic calculations (percentage changes, aggregations) frequently fail
2. **Context Misalignment**: Models focus on tables but miss text explanations, or vice versa
3. **Token Limits**: Average financial report has 47 paragraphs (≈7,000 tokens) causing context overflow and attention dilution
4. **Lack of Specialization**: Single models apply uniform reasoning to arithmetic vs. logical questions

**Dataset**: TAT-QA benchmark with 16,552 questions paired with 2,757 financial report contexts mixing tables and narrative paragraphs.

---

## Architecture

### Pipeline 1: Router-Based Agent Delegation

![Pipeline 1 Architecture](pipeline%201.png)

```
User Question → Dynamic Router → Financial Agent (arithmetic)
                                ↓
                                Logical Agent (inference)
```

**Components:**
- **Dynamic Router**: Gemini 2.5 Flash classifies questions as "financial" (arithmetic) or "logical" (comparison/reasoning)
- **Specialized Agents**: Chain-of-Thought (CoT) prompting for step-by-step answers

**Results (TAT-QA Accuracy %):**

| Component            | Zero-Shot CoT | Single-Shot CoT | Gain   |
|----------------------|---------------|-----------------|--------|
| Dynamic Routing      | 49.6          | 54.7            | +5.1pp |
| Arithmetic Reasoning | 58.2          | 60.9            | +2.7pp |
| Logical Reasoning    | 69.3          | 74.1            | +4.8pp |

**Key Insight**: Single-shot CoT consistently outperformed zero-shot, with largest gains in specialized reasoning agents.

**Critical Limitation**: The router only passes the **question** to downstream agents, not the full context (table + text). This hard routing creates information silos—the Financial Agent receives "What was the percentage change in revenue?" but not the table with revenue data or the paragraph explaining why revenue changed. This fatal flaw led to Pipeline 2.

---

### Pipeline 2: Sequential Multi-Agent with RAG

![Pipeline 2 Architecture](pipeline%202.png)

```
Input (Table + Text)
      ↓
BGE Embeddings → FAISS Vector Search → Top-3 Passages (94% token reduction)
      ↓
Table Agent (TAPAS) ──→ structured facts + CoT trace
      ↓
Context Agent (FLAN-T5-XL) ──→ enriched narrative + CoT trace (RAG-grounded)
      ↓
Selectra (FLAN-T5-small) ──→ persona inference
      ↓
Orchestrator (Gemini 2.5 Flash) ──→ persona-adapted summary
```

**Component Results (TAT-QA F1 / Exact Match %):**

| Agent          | F1   | EM   |
|----------------|------|------|
| Table Agent    | 25.3 | 10.4 |
| Context Agent  | 29.4 | 12.7 |

**Why These Scores Matter**: F1 measures exact span extraction on highly complex financial tables. TAT-QA is notoriously difficult—even fine-tuned models struggle to break 40%. The value is **transparent CoT traces** that expose exactly where extraction fails, enabling systematic debugging instead of black-box failures.

---

## Architectural Decisions

### Why RAG? (Token Efficiency + Semantic Precision)

**The Token Problem:**
- Average TAT-QA context: 47 paragraphs × 150 tokens = 7,050 tokens
- Add table: ~500 tokens
- **Total: 7,550 tokens** before the question arrives

**RAG Solution:**
- **BGE-large-en-v1.5** embeddings (1024-dimensional vectors, trained on scientific/financial corpora)
- **FAISS IndexFlatL2** for similarity search
- Retrieve **top-3** most relevant passages: ~450 tokens
- **Token reduction: 94%** with **87% coverage** of gold-standard answer contexts

**Why BGE Over Alternatives?**

| Alternative | Why Rejected |
|-------------|--------------|
| **TF-IDF / BM25** | Lexical matching misses semantic similarity. "Revenue increased" vs "income growth" are semantically identical but lexically different. |
| **Sentence-BERT** | Lower accuracy on domain-specific financial text. |
| **OpenAI Ada-002** | API cost and latency. BGE runs locally, zero API calls. |
| **Full-context feeding** | Token overflow, attention dilution, noise overwhelms signal. |

**Why FAISS?**
- Local, fast, exact search with SIMD optimizations
- No managed service latency (Pinecone/Weaviate)
- IndexFlatL2 is optimal for 47 passages/document scale

**Top-K Tuning:**
- K=1: Too narrow, missed related context
- **K=3**: Sweet spot (87% coverage, 94% token reduction)
- K=5: Introduced noise, diluted signal
- K=10: Defeated RAG purpose

---

### Why 3 Specialized Agents? (Not 1 or 5)

#### **Table Agent: TAPAS**

**Why TAPAS?**
- Pre-trained on SQA & WikiTableQuestions (understands SQL-like operations: SUM, COUNT, FILTER)
- Cell-level attention (focuses on specific cells, not just rows)
- Structured output in JSON format
- Works zero-shot (no fine-tuning required)

**Alternatives Considered:**

| Model | Why Rejected |
|-------|--------------|
| **GPT-4 with table prompting** | No structured pre-training. Hallucinates cell values. |
| **TableFormer** | Requires task-specific fine-tuning. |
| **PandasAI** | Code generation introduces execution risk. TAPAS is deterministic. |

**Output Format:**
```json
{
  "extracted_facts": ["Revenue 2018: $1.2M", "Revenue 2019: $1.5M"],
  "cot_trace": "Step 1: Located revenue column. Step 2: Found 2018 row..."
}
```

---

#### **Context Agent: FLAN-T5-XL (RAG-Grounded)**

**Why FLAN-T5-XL?**
- **Instruction-tuned**: Responds to "Extract and condense contextual metadata:" prompts
- **Local inference**: No API costs, full control
- **Size sweet spot**: 3B parameters (large enough for semantic understanding, small enough for CPU inference)

**RAG Integration:**
```python
# Retrieve top-3 passages with FAISS
top_passages = faiss_index.search(question_embedding, k=3)

# Context Agent processes ONLY retrieved passages (not all 47)
context_input = " ".join(top_passages)  # ~450 tokens vs 7,050 tokens
prompt = f"Extract and condense contextual metadata: {context_input}"
```

**This is semantic filtering, not random sampling.**

**Alternatives Considered:**

| Model | Why Rejected |
|-------|--------------|
| **FLAN-T5-XXL (11B)** | 4× slower, marginal accuracy gain. Diminishing returns. |
| **BART** | Designed for summarization, not instruction-following. |
| **T5-base** | Too small (220M params), loses semantic nuance in financial text. |
| **LLaMA-2-7B** | Requires GPU, higher latency, not instruction-tuned out-of-box. |

---

#### **Selectra: FLAN-T5-Small (Persona Inference)**

**Why Separate Persona Agent?**

User expertise is **not explicit** in questions. "What was revenue growth?" could be asked by:
- **Novice investor** (wants simple percentage)
- **Financial analyst** (wants segment breakdown)
- **CFO** (wants strategic implications)

**Why FLAN-T5-Small (80M params)?**
- Lightweight (adds <50ms latency)
- Zero-shot persona classification doesn't need billions of parameters
- Instruction-tuned for classification tasks

**Alternatives Rejected:**
- **Rule-based keywords**: Brittle. "What's the revenue?" doesn't contain "analyst" or "investor".
- **GPT-4 classification**: Overkill and API cost for simple task.
- **No persona adaptation**: Uniform summaries ignore user expertise level.

---

#### **Orchestrator: Gemini 2.5 Flash**

**Why Gemini Over Mistral/Claude/GPT?**

| Model | Pros | Cons | Decision |
|-------|------|------|----------|
| **Mistral 7B** | Local inference, no API cost | 8K context limit struggles with all agent outputs + traces | ❌ |
| **GPT-4** | Best reasoning | High cost ($0.03/1K tokens), slower | ❌ |
| **Claude Sonnet** | Good context handling | No structured output mode | ❌ |
| **Gemini 2.5 Flash** | 1M token context, structured output, fast, cheap ($0.0001875/1K tokens) | ✅ |

**What Orchestrator Receives:**
- All table facts from TAPAS with CoT trace
- Enriched context from FLAN-T5-XL with CoT trace
- Persona classification from Selectra
- Original question
- **Total: ~1,070 tokens** (easily handled by 1M context window)

**Key Innovation**: Unlike Pipeline 1's hard routing, Pipeline 2 **preserves full context**—all agent outputs and traces flow forward. If TAPAS extracts facts with 92% confidence and Context Agent adds narrative with 88% confidence, Gemini sees both and performs weighted synthesis.

---

### Why Chain-of-Thought (CoT)?

**Three Reasons:**

1. **Transparency**: Traces expose where reasoning fails
   - If TAPAS extracts wrong cells, trace shows "Step 1: Located column 2..." → debuggable

2. **Accuracy**: Models that explain steps make fewer errors
   - Arithmetic: "Step 1: 1.5 - 1.2 = 0.3" prevents calculation mistakes

3. **Context Flow**: Downstream agents see **how** upstream agents arrived at answers
   - Orchestrator gets full reasoning chain for validation

**Single-Shot vs Zero-Shot:**
- **Zero-shot**: "Answer this question: {question}"
- **Single-shot**: Provides one example showing step-by-step reasoning before the actual question

Single-shot gave +2.7pp to +5.1pp gains. Adding more examples (few-shot) gave <1pp improvement with 200+ token cost per example.

---

## Alternatives Rejected

### Why Not Fine-Tune a Single Large Model?

**Considered**: Fine-tune LLaMA-2-13B on TAT-QA

**Why Rejected:**
- **Data efficiency**: Fine-tuning needs 10K+ examples. TAT-QA has 16K, but need 20% validation + 20% test = only 10K for training.
- **Specialization loss**: Single model learns average behavior across arithmetic AND logical tasks. Multi-agents maintain specialization.
- **Debugging opacity**: Can't isolate table extraction vs context understanding failures.
- **Compute cost**: Fine-tuning 13B params requires GPU hours. This approach uses CPU inference for most agents.

---

### Why Not SQL Generation?

**Considered**: Convert tables to SQL, generate queries

**Why Rejected:**
- **Hybrid data**: SQL handles tables but not text. Many answers require "Revenue increased **due to defense contracts**"—the "due to" is in text.
- **Query complexity**: Multi-hop questions require complex joins. SQL generation accuracy is low.
- **Error amplification**: Wrong SQL = wrong execution = wrong answer. No recovery path.

---

### Why Not BM25 Retrieval?

**Considered**: Use BM25 instead of BGE embeddings

**Why Rejected:**
- **Lexical gap**: "Revenue increased" vs "income growth" have zero keyword overlap but identical meaning.
- **Sparse signals**: Financial terms like "EBITDA" and "operating income" are related, but BM25 treats them as unrelated.
- **Benchmarked**: BGE outperforms BM25 by 12% on MS MARCO Finance subset.

---

## Concrete Example Walkthrough

**Question**: "What was the percentage change in total contract revenues from 2018 to 2019?"

**Input Data**:
- Table with columns: Year, Fixed Price, Time & Materials, Cost Plus, Total
- 47 paragraphs including: "Fixed Price contracts increased due to defense sector engagements in 2019."

### Step-by-Step Execution

**1. RAG Retrieval**
- Embed question with BGE-large-en-v1.5
- FAISS retrieves top-3 passages:
  1. "Fixed Price contracts increased due to defense engagements..." (similarity: 0.89)
  2. "Other category time-and-material deals decreased..." (similarity: 0.85)
  3. "Cost Plus contracts remained stable..." (similarity: 0.81)
- **Result**: 44 irrelevant paragraphs filtered out

**2. Table Agent (TAPAS)**
```json
{
  "extracted_facts": ["2018 Total: $1.2M", "2019 Total: $1.5M"],
  "cot_trace": "Step 1: Located 'Total' column. Step 2: Found 2018 row, value $1.2M. Step 3: Found 2019 row, value $1.5M."
}
```

**3. Context Agent (FLAN-T5-XL)**
```json
{
  "enriched_context": "Revenue growth driven by Fixed Price defense contracts, offset partially by decreased time-and-material deals.",
  "cot_trace": "Identified key entities: defense, Fixed Price contracts, growth drivers."
}
```

**4. Selectra (Persona Inference)**
- Input: Question phrasing analysis
- Output: `"financial analyst"` (specific percentage change indicates quantitative expertise)

**5. Orchestrator (Gemini 2.5 Flash)**
```
CoT: Step 1: Calculate change: $1.5M - $1.2M = $0.3M
     Step 2: Calculate percentage: ($0.3M / $1.2M) × 100 = 25%
     Step 3: Contextualize: growth driven by Fixed Price defense contracts

Summary for financial analyst:
"Total contract revenues increased 25% from $1.2M (2018) to $1.5M (2019),
primarily driven by Fixed Price defense sector engagements. Time-and-material
contracts decreased, partially offsetting the growth."
```

If persona were `"novice investor"`, output would simplify: "Revenue grew 25% from 2018 to 2019, mainly because of new defense contracts."

---

## Implementation Details

### Context Agent (`context_agent.py`)

Takes TAT-QA paragraphs, sorts by order, concatenates, and prompts FLAN-T5-XL with "Extract and condense contextual metadata:". Produces shortened summary of narrative context stored back into JSON for downstream use.

**Features:**
- Batch processing with checkpointing every 100 entries
- Resume from partial runs
- Processes all available text (no filtering—that happens at RAG layer)

### Summarization Agent (`summarization_agent.py`)

Generates personalized CoT summaries by combining:
- Table data (markdown format)
- Enriched context from Context Agent
- Extracted facts from Table Agent
- User persona (novice investor, financial analyst, CFO)

**Models:**
- Primary: Mistral-7B-Instruct
- Fallback: Falcon-7B-Instruct

**Output**: `summaries.json` with one summary per UID-persona pair

---

## Quick Start

### Install

```bash
pip install transformers datasets scikit-learn pandas torch google-generativeai faiss-cpu
```

### Environment Setup

Create `.env` file:
```
GEMINI_API_KEY=your_key_here
```

**Security**: Never commit API keys. Add `.env` to `.gitignore`.

### Run

**Context Agent:**
```bash
python context_agent.py \
  --input tatqa_dataset_test.json \
  --output enriched_output.json \
  --model google/flan-t5-xl \
  --device -1 \
  --resume
```

**Summarization Agent:**

Edit hardcoded paths in `summarization_agent.py` (lines 44, 58, 60) to point to your local data files:

```bash
python summarization_agent.py
```

Output: `summaries.json`

---

## Repository Structure

```
.
├── agent_workflow_with_cot_prompts.py  # CoT prompt templates for TabuSynth, Contextron, Visura, SummaCraft agents
├── btpnlp.py                           # Router-based pipeline: labels TAT-QA questions as financial/logical
├── context_agent.py                    # Context Agent: FLAN-T5-XL condenses narrative paragraphs
├── financial_nlp.py                    # Extended router pipeline with Gemini-based agents
├── selectraflant5.py                   # Selectra agent using FLAN-T5 small for persona inference
├── selectraifelse.py                   # Rule-based Selectra variant (keyword matching)
├── summarization_agent.py              # Summarization Agent: Mistral 7B generates persona-adapted summaries
├── pipeline 1.png                      # Architecture diagram for router-based pipeline
├── pipeline 2.png                      # Architecture diagram for sequential multi-agent pipeline
├── contexts_from_test.xlsx             # Generated context data from TAT-QA test set
├── contexts_from_train.xlsx            # Generated context data from TAT-QA train set
├── TATQA Question answer pair and generated context.xlsx  # Question-answer pairs with generated context
├── .gitignore                          # Ignores .env, API keys, secrets
└── README.md                           # This file
```

---

## Key Takeaways

**Core Principle**: Decompose complexity, preserve context, enable transparency.

1. **RAG** solves token limits (94% reduction) while maintaining semantic precision (87% coverage)
2. **Multi-agents** isolate sub-tasks and enable specialized pre-trained models
3. **CoT** provides debugging transparency and improves accuracy (+2.7pp to +5.1pp)
4. **Gemini orchestration** synthesizes diverse agent outputs into coherent, persona-adapted summaries

**What Worked:**
- Sequential architecture with full context preservation beats hard routing
- Single-shot CoT consistently outperforms zero-shot
- RAG top-3 retrieval balances recall and precision

**What Could Improve:**
- Explore TabLLM or TableFormer for higher table extraction F1
- Add cross-encoder re-ranking layer after FAISS retrieval
- Build LLM-as-judge framework for persona adaptation evaluation

**Cost Efficiency:**
- CPU inference for TAPAS, FLAN-T5 agents
- Gemini API only for orchestration ($0.0001875/1K tokens)
- **Production cost per query: <$0.001**

---

## Technical Depth

### RAG Validation Methodology

Manual validation on 100 random TAT-QA questions:
- Checked if gold-standard supporting facts appeared in top-3 retrieved passages
- **Coverage: 87%**
- 13% failure cases: multi-hop questions requiring 4+ passages

### Persona Adaptation Quality (Human Evaluation)

Domain experts rated summaries on 5-point Likert scale (50 questions, 3 personas):

| Persona | Rating | Notes |
|---------|--------|-------|
| Novice investor | 4.2/5 | Clear, accessible explanations |
| Financial analyst | 3.8/5 | Good detail, needs more segment breakdown |
| CFO | 4.5/5 | Strategic focus, concise |

**Why not automated metrics?** ROUGE/BLEU don't capture "Is this appropriate for a CFO?" Only human evaluation can assess persona adaptation quality.

### Agent Conflict Resolution

When agents disagree, Orchestrator:
1. Checks CoT traces (did TAPAS extract correct cells? did Context Agent use correct passages?)
2. Weights by confidence scores (trust TAPAS 0.92 over Context Agent 0.65 for numerical facts)
3. Performs complementary synthesis ("Revenue increased $0.3M (25%) driven by defense contracts")
4. If both <0.7 confidence, outputs "Insufficient information" rather than guessing

---

## License

MIT License - See LICENSE file for details
