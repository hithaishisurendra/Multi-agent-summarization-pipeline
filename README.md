# Adaptive Agent-Based Personalized Contextual Table Summarization

This system answers questions over hybrid table-and-text data by decomposing the reasoning task across specialized agents. The core problem is that large language models struggle to reason jointly over structured tables and unstructured narrative context, particularly on financial documents that demand multi-step numerical calculations and logical inference.

## Architecture

The project implements two pipelines evaluated on the TAT-QA benchmark, a dataset of 16,552 questions paired with 2,757 financial report contexts mixing tables and descriptive paragraphs.

### Pipeline 1: Router-Based Agent Delegation

![Pipeline 1 Architecture](pipeline%201.png)

```
User Question
      |
      v
Dynamic Router -----> Financial Agent (arithmetic)
      |
      +-------------> Logical Agent (inference)
```

The Dynamic Router classifies incoming questions as financial (arithmetic) or logical (comparison/reasoning). Each specialized agent uses Chain-of-Thought (CoT) prompting to produce step-by-step answers. All components run on Gemini 2.5 Flash.

**Results (TAT-QA, Accuracy %):**

| Component            | Zero-Shot CoT | Single-Shot CoT |
|----------------------|---------------|-----------------|
| Dynamic Routing      | 49.6          | 54.7            |
| Arithmetic Reasoning | 58.2          | 60.9            |
| Logical Reasoning    | 69.3          | 74.1            |

Single-shot CoT consistently outperformed zero-shot. The largest gains appeared in the specialized reasoning agents (+14.6pp arithmetic, +15.9pp logical).

**Limitations:** The router can misclassify hybrid questions. Hard routing discards shared context between table and text, isolating downstream agents from information needed for accurate reasoning.

### Pipeline 2: Sequential Multi-Agent Pipeline

![Pipeline 2 Architecture](pipeline%202.png)

```
Input (Table + Text)
      |
      v
Table Agent (TAPAS) -----> structured facts + CoT trace
      |
      v
Context Agent (FLAN-T5) -> enriched narrative + CoT trace
      |
      v
Summarization Agent (Mistral 7B) -> personalized summary
```

Agents pass both outputs and reasoning traces forward. The Context Agent merges table extractions with text. The Summarization Agent synthesizes all upstream information to generate user-specific summaries. A Selectra module (FLAN-T5 small) infers user type (analyst, manager, novice, investor) from question phrasing to adapt tone and detail level.

**Component Results (TAT-QA, F1 / Exact Match %):**

| Agent          | F1   | EM   |
|----------------|------|------|
| Table Agent    | 25.3 | 10.4 |
| Context Agent  | 29.4 | 12.7 |

The sequential design preserves context across stages. End-to-end quantitative evaluation of the full pipeline was not completed during this phase.

## The Two Agents I Wrote

### Context Agent

The Context Agent (`context_agent.py`) takes the paragraphs field from each TAT-QA entry, sorts them by order, concatenates them into a single string, and prompts FLAN-T5-XL with "Extract and condense contextual metadata:" The model produces a shortened summary of the narrative context. This enriched context is stored back into the JSON for downstream use. The agent processes datasets in batch with checkpointing every 100 entries and supports resume from partial runs. It does not perform filtering or selection. It condenses all available text into a form the summarization agent can consume.

### Summarization Agent

The Summarization Agent (`summarization_agent.py`) generates personalized CoT summaries by combining table data, enriched context from the Context Agent, and extracted facts from a table synthesis step. It builds a prompt that includes the user persona (novice investor, financial analyst, executive summary for CFO), a snippet of the table in markdown format, the enriched context, and the extracted facts. The prompt instructs the model to use chain-of-thought reasoning and produce a concise summary. The agent attempts to load Mistral-7B-Instruct and falls back to Falcon-7B-Instruct if unavailable. It iterates over personas and produces one summary per UID-persona pair. Output is written to `summaries.json`.

## Results

Single-shot CoT prompting improved accuracy over zero-shot CoT across all components of the initial router-based pipeline. The specialized reasoning agents saw the largest gains. The sequential pipeline's Table and Context agents achieved modest F1 scores (25.3% and 29.4% respectively), reflecting the difficulty of structured extraction and context generation on complex financial data. These scores provide benchmarks for component performance but may not fully capture context integration quality, which is central to the approach. The refined pipeline was designed but not evaluated end-to-end with reportable metrics.

**Methodological note from the report:** Direct comparisons to external baselines like TAT-LLM were not performed. Zero-shot performance was modest, partly due to using relatively small models (Gemini 2.5 Flash) without task-specific fine-tuning.

## Quick Start

### Install

```bash
pip install transformers datasets scikit-learn pandas torch google-generativeai
```

### Environment Setup

Create a `.env` file:

```
GEMINI_API_KEY=your_key_here
```

Do not commit API keys. See `.env.example` if provided. Never use real keys in code.

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

Edit hardcoded paths in `summarization_agent.py` (lines 44, 58, 60) to point to your local data files. Then:

```bash
python summarization_agent.py
```

Output is written to `summaries.json`.

## Repository Structure

```
.
├── agent_workflow_with_cot_prompts.py  # CoT prompt templates for TabuSynth, Contextron, Visura, SummaCraft agents (skeleton, send_llm not implemented)
├── btpnlp.py                           # Router-based pipeline: labels TAT-QA questions as financial/logical, evaluates Gemini 2.5 Flash with zero-shot classification
├── context_agent.py                    # Context Agent: FLAN-T5-XL condenses narrative paragraphs into enriched context
├── financial_nlp.py                    # Extended router pipeline: classification, Gemini-based financial/logical agents with CoT prompting
├── selectraflant5.py                   # Selectra agent using FLAN-T5 small to infer user type (analyst, manager, expert, novice, investor) from question
├── selectraifelse.py                   # Rule-based Selectra variant using keyword matching for user type inference
├── summarization_agent.py              # Summarization Agent: Mistral 7B generates personalized CoT summaries from table, context, extracted facts
├── pipeline 1.png                      # Architecture diagram for router-based pipeline
├── pipeline 2.png                      # Architecture diagram for sequential multi-agent pipeline
├── contexts_from_test.xlsx             # Generated context data from TAT-QA test set
├── contexts_from_train.xlsx            # Generated context data from TAT-QA train set
├── contexts_from_train (1).xlsx        # Duplicate or variant of train contexts
├── TATQA Question answer pair and generated context.xlsx  # Question-answer pairs with generated context
├── BeyondThePrompt_Final_Report (1).pdf  # Final project report (source of all metrics)
├── Final_Report.pdf                    # Duplicate of final report
├── Intermediate_Progress_Report.pdf    # Mid-project progress report
├── Project Proposal.pdf                # Initial project proposal
├── .gitignore                          # Ignores .env, API keys.txt, *.key
└── README.md                           # This file
```
