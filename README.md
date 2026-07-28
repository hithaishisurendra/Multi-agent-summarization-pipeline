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

**Why BGE **: Semantic embeddings capture that "revenue increased" and "income growth" have identical meaning despite zero keyword overlap. BGE trained on financial corpora outperforms others.

**Why K=3**: Tested K=1, 3, 5, 10. K=3 is the sweet spot:

| K     | Coverage | Token Count | Issue                             |
| ----- | -------- | ----------- | --------------------------------- |
| 1     | 62%      | 150         | Too narrow, missed context        |
| **3** | **87%**  | **450**     | **Optimal balance**               |
| 5     | 91%      | 750         | Introduced noise, confused agents |
| 10    | 94%      | 1,500       | Defeats RAG purpose               |

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

**Output**:

```json
{
  "summary": "Total contract revenues increased 25% from $1.2M (2018) to $1.5M (2019), primarily driven by Fixed Price defense sector engagements.",
  "cot_reasoning": "Step 1: Calculate change: $1.5M - $1.2M = $0.3M\nStep 2: Calculate percentage: ($0.3M / $1.2M) × 100 = 25%...",
  "persona": "financial analyst"
}
```

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

| Component     | Model         | Latency     | Cost          |
| ------------- | ------------- | ----------- | ------------- |
| RAG Retrieval | BGE + FAISS   | 55ms        | $0 (local)    |
| Table Agent   | TAPAS         | 300ms       | $0 (local)    |
| Context Agent | FLAN-T5-XL    | 800ms       | $0 (local)    |
| Persona Agent | FLAN-T5-Small | 50ms        | $0 (local)    |
| Orchestrator  | Gemini Flash  | 500ms       | $0.000275     |
| **TOTAL**     |               | **1,705ms** | **$0.000275** |

**Bottleneck**: FLAN-T5-XL Context Agent at 800ms (47% of total latency)

### Scaling to Production (1M queries/month)

**Current Architecture**:

- Cost: $275/month (Gemini only)
- Latency: 1.7 seconds

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

| Operation                  | Time                        | Notes                  |
| -------------------------- | --------------------------- | ---------------------- |
| Model loading              | 2-3 min                     | One-time on first run  |
| RAG index (10 examples)    | 30 sec                      | BGE embeddings + FAISS |
| RAG index (full dataset)   | 25 min                      | 2,757 contexts         |
| Single question (with RAG) | 1.7 sec                     | All 5 agents           |
| Single question (no RAG)   | Would exceed context limits | Not viable             |

---

## Validation Methodology

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

---

## Citation

If you use this work, please cite:

```
Multi-Agent Table Question-Answering with RAG
TAT-QA Benchmark Implementation
2025
```

**Dataset**: TAT-QA (Zhu et al., 2021) - https://github.com/NExTplusplus/TAT-QA

---

## License

MIT License - See LICENSE file for details
