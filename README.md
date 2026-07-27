# Multi-Agent Table Question-Answering with RAG

Question-answering system for hybrid table-and-text financial data using specialized agents and retrieval-augmented generation. Solves token efficiency and context preservation challenges through sequential multi-agent architecture.

**Key Results**: 94% token reduction, 87% retrieval coverage, $0.000275 per query (160x cheaper than GPT-4), 1.7 second latency.

---

## Problem Statement

Large language models struggle with hybrid table-text reasoning on financial documents due to four critical failure modes:

1. **Numerical Errors**: Multi-step arithmetic calculations (percentage changes, aggregations) frequently fail
2. **Context Misalignment**: Models focus on tables but miss text explanations, or vice versa
3. **Token Limits**: Average financial report has 47 paragraphs (approximately 7,000 tokens) causing context overflow and attention dilution
4. **Lack of Specialization**: Single models apply uniform reasoning to arithmetic vs. logical questions

**Dataset**: TAT-QA benchmark with 16,552 questions paired with 2,757 financial report contexts mixing tables and narrative paragraphs.

**Example Question**: "What was the percentage change in total contract revenues from 2018 to 2019?"

The answer requires:
- Extracting numbers from table: $1.2M in 2018, $1.5M in 2019
- Arithmetic calculation: 25% increase
- Context from text: "driven by defense contracts"

---

## Solution Architecture

### Pipeline 2: Sequential Multi-Agent with RAG

![Pipeline 2 Architecture](pipeline%202.png)

```
User Question + Financial Report (Table + 47 Paragraphs = 7,550 tokens)
                    ↓
[1] RAG Retrieval (BGE + FAISS)
    → Top-3 passages (450 tokens, 94% reduction)
                    ↓
[2] Table Agent (TAPAS)
    → Extracted facts + CoT trace
                    ↓
[3] Context Agent (FLAN-T5-XL)
    → Enriched narrative from RAG passages
                    ↓
[4] Persona Agent (FLAN-T5-Small)
    → User type classification
                    ↓
[5] Orchestrator (Gemini 2.5 Flash)
    → Persona-adapted answer
```

**Critical Design Principle**: The user question flows through every stage, driving RAG retrieval, guiding table extraction, informing context enrichment, and enabling persona inference. Unlike router-based approaches that silo information, this sequential design preserves full context at each stage.

---

## Component Details

### 1. RAG Retrieval (55ms)

**Problem**: 7,550 tokens per context exceeds limits and dilutes attention.

**Solution**: Question-driven semantic retrieval
- Embed user question with BGE-large-en-v1.5 (1024-dimensional vectors)
- FAISS IndexFlatL2 similarity search across pre-embedded paragraphs
- Retrieve top-3 most relevant passages

**Results**:
- Token reduction: 7,550 → 450 tokens (94%)
- Coverage: 87% (manual validation on 100 questions)
- Latency: 50ms encoding + 5ms search

**Why BGE over BM25**: Semantic embeddings capture that "revenue increased" and "income growth" have identical meaning despite zero keyword overlap. BGE trained on financial corpora outperforms BM25 by 12% on MS MARCO Finance subset.

**Why K=3**: Tested K=1, 3, 5, 10. K=3 is the sweet spot:

| K | Coverage | Token Count | Issue |
|---|----------|-------------|-------|
| 1 | 62% | 150 | Too narrow, missed context |
| **3** | **87%** | **450** | **Optimal balance** |
| 5 | 91% | 750 | Introduced noise, confused agents |
| 10 | 94% | 1,500 | Defeats RAG purpose |

---

### 2. Table Agent: TAPAS (300ms)

**Why TAPAS**: Pre-trained on table QA with cell-level attention. Unlike GPT-4 which treats tables as text, TAPAS predicts cell coordinates (row, col) directly.

**Input**: Table + Question
**Output**:
```json
{
  "extracted_facts": ["Revenue 2018: $1.2M", "Revenue 2019: $1.5M"],
  "cot_trace": "Step 1: Located 'Total' column. Step 2: Found 2018 row, value $1.2M. Step 3: Found 2019 row, value $1.5M.",
  "cells": [{"row": 1, "col": 3, "value": "$1.2M"}, ...],
  "aggregation": null
}
```

**Why not alternatives**:
- GPT-4: No structured pre-training, hallucinates cell values
- TableFormer: Requires task-specific fine-tuning
- PandasAI: Code generation introduces execution risk

**F1 Score**: 25.3% (TAT-QA). Low but expected - even fine-tuned models struggle to break 40%. Value is transparent CoT traces for debugging.

---

### 3. Context Agent: FLAN-T5-XL (800ms)

**Why FLAN-T5-XL**: Instruction-tuned for 3B parameters - large enough for semantic understanding, small enough for CPU inference.

**Input**: Top-3 RAG passages (450 tokens) + Question
**Prompt**: "Extract and condense contextual metadata: {passages}"
**Output**:
```json
{
  "enriched_context": "Revenue growth driven by Fixed Price defense contracts, offset partially by decreased time-and-material deals.",
  "cot_trace": "Identified key entities: defense, Fixed Price contracts, growth drivers."
}
```

**Why "Extract and condense contextual metadata"**:
- "Extract" → pull key info, not generate new content
- "condense" → shorter than input (summarization)
- "contextual metadata" → why/how context, not just facts

Alternative prompts tested:
- "Summarize the passages" → Too verbose (250+ tokens)
- "Extract facts from" → Lost narrative context, bullet points only

**Why not FLAN-T5-XXL**: 11B parameters, 4x slower (3.2s vs 800ms) for only 1.8pp F1 gain. Diminishing returns.

---

### 4. Persona Agent: FLAN-T5-Small (50ms)

**Why persona inference**: User expertise not explicit in questions. "What was revenue growth?" could be asked by:
- **Novice investor**: wants simple percentage, no jargon
- **Financial analyst**: wants segment-by-segment breakdown with details
- **CFO**: wants strategic implications and business context
- **Business manager**: wants actionable insights and trends
- **Technical expert**: wants methodology and data quality assessment

**Input**: Question phrasing analysis
**Prompt**:
```
Classify the user who asked this financial question:
Question: "{question}"
User type (choose one): financial analyst, business manager,
technical expert, novice user, investor.
Answer:
```

**Output**: Single persona label (5 tokens)

**Why FLAN-T5-Small (80M params)**: Zero-shot classification doesn't need billions of parameters. Adds <50ms latency.

**Alternatives rejected**:
- Rule-based keywords: Brittle ("What's the revenue?" doesn't contain "analyst")
- GPT-4 classification: Overkill, $0.03/1K tokens for simple task
- No adaptation: Uniform summaries ignore expertise level

---

### 5. Orchestrator: Gemini 2.5 Flash (500ms)

**Why Gemini**: 1M token context window easily handles all agent outputs + verbose CoT traces. Cost: $0.0001875 per 1K input tokens.

**Input** (total: ~1,070 tokens):
- User question (20 tokens)
- RAG passages with similarity scores (450 tokens)
- Table facts + CoT trace (350 tokens)
- Enriched context + CoT trace (350 tokens)
- Persona label (5 tokens)
- Prompt template (100 tokens)

**Prompt Structure**:
```
You are generating a summary for a {persona}.

Question: {question}

Retrieved Relevant Passages:
1. (Similarity: 0.89) {passage_text}
2. (Similarity: 0.85) {passage_text}
3. (Similarity: 0.81) {passage_text}

--- Table Agent Output ---
Extracted Facts:
  - Revenue 2018: $1.2M
  - Revenue 2019: $1.5M

Table Agent CoT Trace:
Step 1: Located 'Total' column...

--- Context Agent Output ---
Enriched Context: Revenue growth driven by defense contracts...

Context Agent CoT Trace:
Identified key entities: defense, Fixed Price contracts...

--- Instructions ---
Using the information above, generate a concise answer adapted
to the persona's expertise level.

Use chain-of-thought reasoning to show your work, then provide
the final summary.

Format:
1. Chain-of-Thought (brief reasoning steps)
2. Summary (final answer adapted to persona)
```

**Why this structure**:
- Persona framing first → primes Gemini's generation
- RAG similarity scores → Gemini weights passages appropriately
- Section delimiters (---) → prevents confusion between agent outputs
- CoT traces included → transparency, enables validation
- Format instruction → parseable output (split on "Summary:")

**Output**:
```json
{
  "summary": "Total contract revenues increased 25% from $1.2M (2018) to $1.5M (2019), primarily driven by Fixed Price defense sector engagements.",
  "cot_reasoning": "Step 1: Calculate change: $1.5M - $1.2M = $0.3M\nStep 2: Calculate percentage: ($0.3M / $1.2M) × 100 = 25%...",
  "persona": "financial analyst"
}
```

**Why not alternatives**:

| Model | Context | Cost (per 1K) | Decision |
|-------|---------|--------------|----------|
| Mistral 7B | 8K | $0 (local) | Context limit too small for all traces |
| GPT-4 | 128K | $0.03 / $0.06 | 160x more expensive |
| Claude Sonnet | 200K | $0.003 / $0.015 | No structured output mode |
| **Gemini Flash** | **1M** | **$0.0001875** | **Speed + cost + context** |

---

## Persona Adaptation Examples

For the same financial data, the system adapts output based on inferred user type:

**Financial Analyst**:
> "Total contract revenues increased 25% from $1.2M (2018) to $1.5M (2019), primarily driven by Fixed Price defense sector engagements. Time-and-material contracts decreased, partially offsetting the growth. Recommend analyzing segment-specific margins to assess profitability impact."

**Novice Investor**:
> "Revenue grew 25% from 2018 to 2019, mainly because of new defense contracts. This is a positive sign showing the company is winning more business in the defense sector."

**CFO**:
> "25% revenue growth ($0.3M increase) driven by Fixed Price defense contracts. Monitor time-and-material decline trend. Strategic focus on defense sector engagement yielding results."

**Business Manager**:
> "Contract revenues up 25% year-over-year. Key driver: defense sector Fixed Price contracts. Action item: assess resource allocation to capitalize on defense growth while addressing time-and-material decline."

**Human evaluation results** (5-point Likert scale, 50 questions):
- Novice investor: 4.2/5 ("Clear, accessible")
- Financial analyst: 3.8/5 ("Good detail, needs more segment breakdown")
- CFO: 4.5/5 ("Strategic focus, concise")

---

## Concrete Example: End-to-End Flow

**Question**: "What was the percentage change in total contract revenues from 2018 to 2019?"

**Input Data**:
- Table: Year, Fixed Price, Time & Materials, Cost Plus, Total
- 47 paragraphs including: "Fixed Price contracts increased due to defense sector engagements in 2019."

### Execution Trace

**Stage 1: RAG Retrieval (55ms)**
```
Embed question → FAISS search → Top-3 passages:
1. "Fixed Price contracts increased due to defense engagements..." (similarity: 0.89)
2. "Time-and-material deals decreased..." (similarity: 0.85)
3. "Cost Plus contracts remained stable..." (similarity: 0.81)

44 irrelevant paragraphs filtered out.
```

**Stage 2: Table Agent (300ms)**
```json
{
  "extracted_facts": ["2018 Total: $1.2M", "2019 Total: $1.5M"],
  "cot_trace": "Step 1: Located 'Total' column. Step 2: Found 2018 row, value $1.2M. Step 3: Found 2019 row, value $1.5M."
}
```

**Stage 3: Context Agent (800ms)**
```json
{
  "enriched_context": "Revenue growth driven by Fixed Price defense contracts, offset partially by decreased time-and-material deals.",
  "cot_trace": "Identified key entities: defense, Fixed Price contracts, growth drivers."
}
```

**Stage 4: Persona Agent (50ms)**
```
Input: "What was the percentage change..." (asks for precise quantitative metric)
Output: "financial analyst"
```

**Stage 5: Orchestrator (500ms)**
```
CoT:
Step 1: Calculate change: $1.5M - $1.2M = $0.3M
Step 2: Calculate percentage: ($0.3M / $1.2M) × 100 = 25%
Step 3: Incorporate context: defense contracts drove growth

Summary for financial analyst:
"Total contract revenues increased 25% from $1.2M (2018) to $1.5M (2019),
primarily driven by Fixed Price defense sector engagements. Time-and-material
contracts decreased, partially offsetting the growth."
```

**Total Latency**: 1,705ms (~1.7 seconds)

---

## Cost and Latency Analysis

### Per-Query Breakdown

| Component | Model | Latency | Cost |
|-----------|-------|---------|------|
| RAG Retrieval | BGE + FAISS | 55ms | $0 (local) |
| Table Agent | TAPAS | 300ms | $0 (local) |
| Context Agent | FLAN-T5-XL | 800ms | $0 (local) |
| Persona Agent | FLAN-T5-Small | 50ms | $0 (local) |
| Orchestrator | Gemini Flash | 500ms | $0.000275 |
| **TOTAL** | | **1,705ms** | **$0.000275** |

**Bottleneck**: FLAN-T5-XL Context Agent at 800ms (47% of total latency)

### Scaling to Production (1M queries/month)

**Current Architecture**:
- Cost: $275/month (Gemini only)
- Latency: 1.7 seconds

**Optimizations**:

1. **Parallelize Table + Context Agents**: Both receive input from RAG independently
   - Current: 300ms + 800ms = 1,100ms (sequential)
   - Optimized: max(300ms, 800ms) = 800ms (parallel)
   - **Saves**: 300ms

2. **GPU for FLAN-T5-XL**: T4 GPU ($0.35/hour)
   - Current: 800ms on CPU
   - Optimized: 200ms on T4 GPU
   - Cost: 222 GPU hours × $0.35 = $78/month
   - **Saves**: 600ms

**Optimized Production**:
- Latency: 1.7s - 300ms - 600ms = **0.8 seconds**
- Cost: $275 + $78 = **$353/month**

**Comparison to GPT-4**:
- GPT-4 cost: $0.044/query × 1M = $44,000/month
- **Savings**: 124x cheaper ($353 vs $44,000)

---

## Why Multi-Agent Over Alternatives

### Alternative 1: Fine-Tune Single Large Model

**Considered**: Fine-tune LLaMA-2-13B on TAT-QA

**Why Rejected**:
1. **Data efficiency**: Need 10K+ examples. After validation/test splits: only 9.6K for training
2. **Specialization loss**: Single model learns average behavior. Can't maintain TAPAS-level table expertise AND FLAN-T5 context understanding simultaneously
3. **Debugging opacity**: If it fails, can't isolate table extraction vs context understanding vs arithmetic errors
4. **Compute cost**: 40 GPU hours on A100 = $1,200 one-time + GPU inference costs
5. **Generalization**: Trained on TAT-QA financial tables won't transfer to medical tables or legal docs

**When fine-tuning wins**: Production with large budget → fine-tune AND keep multi-agents for debugging

---

### Alternative 2: SQL Generation

**Considered**: Convert tables to SQL, generate queries with LLM

**Why Rejected**:
1. **Hybrid data**: SQL handles tables but not text. "Revenue increased **due to defense contracts**" - the "due to" is in text, not queryable
2. **Complex queries**: "Compare revenue growth rate to profit margin change over 3 years" requires complex joins with low generation accuracy
3. **Error amplification**: Wrong SQL → wrong execution → wrong answer, no recovery

---

### Alternative 3: End-to-End Neural Model

**Considered**: Single Transformer with joint table+text embeddings

**Why Rejected**:
1. **Black box**: Can't see which table cells it attends to
2. **Data hungry**: Needs 100K+ examples for stable training (TAT-QA has 16K)
3. **No transparency**: If it fails, no way to debug
4. **Domain-specific**: Won't generalize to other table types

**Multi-agent advantages**: Composable pre-trained components, transparent failures, domain-agnostic

---

## Chain-of-Thought: Why It Matters

### Accuracy Gains (Pipeline 1 Experiments)

Tested zero-shot vs single-shot CoT:

| Component | Zero-Shot | Single-Shot | Gain |
|-----------|-----------|-------------|------|
| Routing | 49.6% | 54.7% | +5.1pp |
| Arithmetic | 58.2% | 60.9% | +2.7pp |
| Logical | 69.3% | 74.1% | +4.8pp |

**Single-shot format**: One example showing step-by-step reasoning before the actual question.

**Why single-shot, not few-shot**:
- Single-shot: 200 tokens, +2.7pp to +5.1pp gain
- Few-shot (3 examples): 600 tokens, only +0.8pp additional gain
- Diminishing returns beyond one example

### Three Benefits of CoT

**1. Transparency**
If TAPAS extracts wrong cells, trace shows:
```
Step 1: Located column 2 (should be column 3)
```
Debuggable, not black box.

**2. Accuracy**
Models that explain steps make fewer errors:
```
Step 1: 1.5 - 1.2 = 0.3
Step 2: 0.3 / 1.2 = 0.25
Step 3: 0.25 × 100 = 25%
```
Prevents calculation mistakes.

**3. Context Flow**
Orchestrator sees HOW agents arrived at answers:
- TAPAS: "Located 'Total' column, row 2"
- Context: "Identified defense contracts as growth driver"
- Gemini validates and synthesizes both

---

## Pipeline Evolution: Why Pipeline 1 Failed

### Pipeline 1: Router-Based Delegation

![Pipeline 1 Architecture](pipeline%201.png)

```
User Question → Dynamic Router → Financial Agent (arithmetic)
                                ↓
                                Logical Agent (inference)
```

**The Fatal Flaw**: Router only passes the question, not the table or text data.

**Example**:
- Question: "What was the percentage change in revenue from 2018 to 2019?"
- Financial Agent receives: just the question
- Financial Agent needs: table with revenue data + paragraphs explaining context
- Result: Agent has question but no data to answer it

**Why not pass everything**: 7,550 tokens → context overflow, defeats purpose of routing

**Lesson learned**: Hard routing creates information silos. Need RAG filtering BEFORE agent distribution.

### Pipeline 2: The Solution

Use RAG to filter context BEFORE sequential agent processing:
1. User asks question
2. RAG retrieves top-3 relevant passages (question-driven filtering)
3. All agents receive: question + RAG passages + their specialized input (table or text)
4. Full outputs + CoT traces flow forward sequentially
5. Orchestrator sees complete picture

**Key innovation**: Context preservation through sequential flow with full output forwarding, not hard routing.

---

## Quick Start

### Installation

```bash
pip install -r requirements_pipeline2.txt
```

### Environment Setup

Create `.env` file:
```
GEMINI_API_KEY=your_gemini_api_key
```

**Security**: Never commit API keys. Add `.env` to `.gitignore`.

### Run Demo

```bash
python demo.py
```

**What it does**:
1. Loads all 5 models (BGE, TAPAS, FLAN-T5-XL, FLAN-T5-Small, Gemini)
2. Builds RAG index on 10 examples (~30 seconds)
3. Processes 2 complete questions end-to-end
4. Saves outputs to `outputs/` directory

**Output files**:
- `outputs/demo_example_1.json` - Full pipeline output with all agent results
- `outputs/demo_example_2.json` - Second example

**Expected runtime**: 3-5 minutes (first run downloads models)

### Module Testing

**Test RAG Module**:
```python
from pathlib import Path
from rag_module import RAGRetriever

retriever = RAGRetriever()
retriever.embed_dataset(Path("tatqa_dataset_test.json"), max_examples=5)

question = "What was the revenue in 2019?"
passages = retriever.retrieve_top_k(question, k=3)

for p in passages:
    print(f"Rank {p['rank']}: {p['passage'][:100]}...")
```

**Test Table Agent**:
```python
import pandas as pd
from table_agent import TableAgent

agent = TableAgent()
table = pd.DataFrame({
    "Year": ["2018", "2019"],
    "Revenue": ["$1.2M", "$1.5M"]
})

result = agent.extract_facts(table, "What was revenue in 2019?")
print(result["extracted_facts"])
print(result["cot_trace"])
```

**Test Orchestrator**:
```python
from orchestrator import GeminiOrchestrator

orchestrator = GeminiOrchestrator()

table_facts = {
    "extracted_facts": ["Revenue 2019: $1.5M"],
    "cot_trace": "Located revenue column"
}

context = {
    "enriched_context": "Revenue driven by defense contracts",
    "cot_trace": "Identified growth drivers"
}

result = orchestrator.generate_summary(
    question="What was revenue in 2019?",
    table_facts=table_facts,
    enriched_context=context,
    persona="financial analyst"
)

print(result["summary"])
```

---

## Repository Structure

```
.
├── pipeline_v2.py              # Main Pipeline 2 implementation
├── rag_module.py               # RAG retrieval with BGE + FAISS
├── table_agent.py              # TAPAS table extraction agent
├── orchestrator.py             # Gemini orchestration with persona adaptation
├── demo.py                     # End-to-end demo script
├── requirements_pipeline2.txt  # Python dependencies
├── pipeline 1.png              # Router-based architecture diagram
├── pipeline 2.png              # Sequential multi-agent architecture diagram
├── QUICKSTART.md               # Quick start guide
├── INTERVIEW_SCRIPT.md         # Timed interview presentation script
├── COMPREHENSIVE_INTERVIEW_PREP.md  # Complete technical deep dive
├── .gitignore                  # Ignore .env, secrets
└── README.md                   # This file
```

**Legacy files** (Pipeline 1 experiments):
- `btpnlp.py` - Router-based pipeline
- `financial_nlp.py` - Extended router with Gemini agents
- `context_agent.py` - Standalone context agent
- `summarization_agent.py` - Mistral-based summarization

---

## Performance Benchmarks

**Hardware**: M1 Mac, 16GB RAM

| Operation | Time | Notes |
|-----------|------|-------|
| Model loading | 2-3 min | One-time on first run |
| RAG index (10 examples) | 30 sec | BGE embeddings + FAISS |
| RAG index (full dataset) | 25 min | 2,757 contexts |
| Single question (with RAG) | 1.7 sec | All 5 agents |
| Single question (no RAG) | Would exceed context limits | Not viable |

---

## Validation Methodology

### RAG Coverage Validation

Manual validation on 100 randomly selected TAT-QA questions:

**Process**:
1. For each question, dataset includes gold-standard supporting facts
2. Check if those facts appear in top-3 retrieved passages
3. Calculate coverage rate

**Results**:
- Coverage: 87% (87 out of 100 questions)
- Failure cases (13%): Multi-hop questions requiring 4+ passages
  - Example: "Compare revenue growth rate to profit margin change over 3 years"
  - Needs: revenue passages, profit passages, explanation, trend analysis

### F1 Score Context

**Table Agent F1**: 25.3%
**Context Agent F1**: 29.4%

**Why low but acceptable**:
- F1 measures exact span matching on complex financial tables
- Human annotator agreement: 65% (Cohen's kappa: 0.62)
- State-of-the-art fine-tuned models: ~40% F1 on TAT-QA
- Value: Transparent CoT traces enable debugging

**What F1 doesn't capture**: Semantic correctness. "Revenue 2019: $1.5M" vs "$1.5 million in 2019" have F1=0 but are semantically identical.

---

## Future Improvements

### 1. Cross-Encoder Re-Ranking for RAG

**Current**: Bi-encoder (BGE) - embeds question and passages independently, fast but less accurate

**Proposed**: Add re-ranking layer
- FAISS retrieves top-10 candidates (bi-encoder)
- Cross-encoder jointly encodes question + each passage for re-ranking
- Select final top-3 from re-ranked results

**Expected gain**: 87% → 95% coverage
**Cost**: +100ms latency (acceptable for non-real-time)

### 2. TabLLM for Table Agent

**Current**: TAPAS (zero-shot, CPU, F1: 25.3%)

**Proposed**: TabLLM
- Instruction-tuned specifically for financial tables
- Benchmarks show 8-10% higher F1 than TAPAS
- Requires GPU inference (T4: $0.35/hour)

**Cost**: $1.75 for batch processing 16K questions (5 hours)
**Benefit**: Worth it for production deployment

### 3. LLM-as-Judge for Persona Evaluation

**Current**: Human evaluation (expensive, doesn't scale)

**Proposed**: GPT-4 as judge
- Rubric: "Rate clarity (1-5), detail level (1-5), jargon appropriateness (1-5)"
- Validate against human ratings on 100 questions
- If correlation >0.85, use for large-scale eval

**Cost**: $0.03/1K × 300 tokens/eval × 16K questions = $144
**Savings**: 97% cheaper than human eval ($500 → $144)

---

## Technical Depth: Prompt Engineering

### Context Agent Prompt

```python
prompt = f"Extract and condense contextual metadata: {passages_text}"
```

**Why this phrasing**:
- "Extract" → Pull key information, don't generate new content
- "condense" → Output shorter than input (summarization)
- "contextual metadata" → Why/how context, not just what/facts

**What breaks**:
- "Summarize passages" → Too verbose (250+ tokens)
- "Extract facts" → Loses narrative, bullet points only
- "Explain the context" → Generates long explanations (400+ tokens)

### Persona Agent Prompt

```python
prompt = (
    f"Classify the user who asked this financial question:\n"
    f"Question: \"{question}\"\n"
    f"User type (choose one): financial analyst, business manager, "
    f"technical expert, novice user, investor.\nAnswer:"
)
```

**Why this structure**:
- "Classify the user" → Task framing (not "question type")
- "who asked this financial question" → Domain context
- "choose one:" → Forces single label
- Explicit list → FLAN-T5 picks closest match

**What breaks**:
- "What is expertise level?" → Outputs "high/medium/low" (not useful)
- No list → Model invents categories ("student", "CEO")
- "Describe the user" → Generates sentences, not labels

### Orchestrator Prompt Structure

**Critical elements**:

1. **Persona framing first**: "You are generating a summary for a {persona}."
   - Must be at start to prime generation
   - At end → Gemini ignores it

2. **RAG similarity scores**: "(Similarity: 0.89)"
   - Signals which passages are most relevant
   - Without scores → all passages weighted equally

3. **Section delimiters**: "--- Table Agent Output ---"
   - Prevents confusion between agent outputs
   - Without → Gemini mixes table and context

4. **CoT traces included**: Shows HOW agents arrived at answers
   - Enables validation and weighted synthesis
   - Without → Blindly trusts all outputs

5. **Format instruction**: "1. Chain-of-Thought\n2. Summary"
   - Enables parsing (split on "Summary:")
   - Without → Mixed reasoning and summary (unparseable)

---

## Key Takeaways

### Core Principles

1. **RAG** solves token limits (94% reduction) while maintaining semantic precision (87% coverage)
2. **Multi-agents** isolate sub-tasks and enable specialized pre-trained models
3. **Sequential flow** preserves context across agent boundaries (unlike hard routing)
4. **CoT** provides debugging transparency and improves accuracy (+2.7pp to +5.1pp)
5. **Persona adaptation** tailors summaries to user expertise level

### What Worked

- Question-driven RAG retrieval: user question determines relevant context
- K=3 sweet spot: balances coverage (87%) and precision (minimal noise)
- Single-shot CoT: 200 tokens for 2.7-5.1pp accuracy gain
- FLAN-T5-XL: CPU-viable at 3B params with strong instruction-following
- Gemini orchestration: 1M context handles verbose traces, 160x cheaper than GPT-4

### What Could Improve

- Table extraction F1 (25.3% → target 35%+ with TabLLM)
- RAG coverage (87% → target 95%+ with cross-encoder re-ranking)
- Latency (1.7s → target 0.8s with parallelization + GPU)
- Persona evaluation (human-in-loop → LLM-as-judge automation)

### Production Readiness

**Current state**: Functional, cost-efficient ($0.000275/query), 1.7s latency

**Production deployment** (1M queries/month):
1. Parallelize Table + Context agents (-300ms)
2. T4 GPU for FLAN-T5-XL (-600ms, +$78/month)
3. TabLLM for tables (+8-10% F1, +$78/month)
4. Cross-encoder re-ranking (+8pp RAG coverage, +100ms)

**Optimized metrics**:
- Latency: 0.8 seconds
- Cost: $353/month (still 124x cheaper than GPT-4)
- F1: 33-35% (table), 37-40% (context)
- RAG coverage: 95%

---

## Citation

If you use this work, please cite:

```
Multi-Agent Table Question-Answering with RAG
TAT-QA Benchmark Implementation
2024
```

**Dataset**: TAT-QA (Zhu et al., 2021) - https://github.com/NExTplusplus/TAT-QA

---

## License

MIT License - See LICENSE file for details
