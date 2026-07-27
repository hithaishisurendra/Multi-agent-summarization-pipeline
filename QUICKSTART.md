# Pipeline 2 Quick Start Guide

Get the full RAG-augmented multi-agent pipeline running in 10 minutes.

## Prerequisites

- Python 3.8+
- 16GB RAM minimum
- Gemini API key ([get one free](https://makersuite.google.com/app/apikey))
- TAT-QA dataset ([download](https://github.com/NExTplusplus/TAT-QA))

## Installation (5 minutes)

### Option 1: Automated Setup

```bash
# Run the setup script
./setup_and_test.sh
```

### Option 2: Manual Setup

```bash
# Install dependencies
pip install -r requirements_pipeline2.txt

# Set API key
export GEMINI_API_KEY='your-api-key-here'

# Or create .env file
echo "GEMINI_API_KEY=your-api-key-here" > .env
```

## Quick Test (5 minutes)

### Run Demo with 2 Examples

```bash
# Make sure tatqa_dataset_test.json is in current directory
python3 demo.py
```

This will:
1. Load all 5 models (BGE, TAPAS, FLAN-T5-XL, FLAN-T5-Small, Gemini)
2. Build RAG index on 10 examples
3. Process 2 complete examples end-to-end
4. Save outputs to `outputs/` directory

**Expected runtime:** 3-5 minutes (first run downloads models)

### Output

You'll see:
- RAG retrieved passages with similarity scores
- TAPAS extracted table facts
- Context Agent enriched summaries
- Persona classification
- Final orchestrated summary

Files created:
- `outputs/demo_example_1.json`
- `outputs/demo_example_2.json`

## Architecture Overview

```
Input (Table + Paragraphs + Question)
        ↓
[1] RAG Retriever (BGE + FAISS)
    → Top-3 passages (94% token reduction)
        ↓
[2] Table Agent (TAPAS)
    → Extracted facts + CoT trace
        ↓
[3] Context Agent (FLAN-T5-XL)
    → Enriched context from RAG passages
        ↓
[4] Persona Agent (FLAN-T5-Small)
    → User type classification
        ↓
[5] Orchestrator (Gemini 2.5 Flash)
    → Final persona-adapted summary
```

## Scaling to Full Dataset

### Process All TAT-QA Examples

```python
from pathlib import Path
from pipeline_v2 import Pipeline2

# Initialize
pipeline = Pipeline2()

# Build full RAG index (takes ~30 min for 2,757 contexts)
pipeline.build_rag_index(Path("tatqa_dataset_test.json"))

# Process all questions
with open("tatqa_dataset_test.json", "r") as f:
    data = json.load(f)

results = []
for entry in data:
    for q in entry["questions"]:
        result = pipeline.process_single_question(entry, q["question"])
        results.append(result)

# Save results
with open("full_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

**Estimated time for full dataset:** 4-6 hours on CPU

## Module-by-Module Testing

### Test RAG Module

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

### Test Table Agent

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

### Test Orchestrator

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

## Troubleshooting

### Models Won't Load

**Error:** `Out of memory`

**Solution:** Use smaller models:
```python
pipeline = Pipeline2(
    context_model="google/flan-t5-base",  # Instead of XL
    table_model="google/tapas-tiny"       # Instead of base
)
```

### RAG Index Build Fails

**Error:** `FAISS dimension mismatch`

**Solution:** Clear any cached index files and rebuild

### Gemini API Errors

**Error:** `API key not found`

**Solution:**
```bash
export GEMINI_API_KEY='your-key-here'
# Or add to ~/.bashrc for persistence
```

### TAPAS Table Parsing Issues

**Error:** `Table format not recognized`

**Solution:** TAT-QA tables should be in format:
```json
{
  "table": [
    ["Year", "Revenue"],  // Header row
    ["2019", "$1.5M"]     // Data rows
  ]
}
```

## Performance Benchmarks

Hardware: M1 Mac, 16GB RAM

| Operation | Time | Notes |
|-----------|------|-------|
| Model loading | 2-3 min | One-time on first run |
| RAG index (10 examples) | 30 sec | Embeddings + FAISS |
| RAG index (full dataset) | 25 min | 2,757 contexts |
| Single question (with RAG) | 8-12 sec | All agents |
| Single question (no RAG) | 15-20 sec | All paragraphs |

## What's Next?

- Run demo: `python3 demo.py`
- Process custom questions: Modify `demo.py` with your own examples
- Scale to full dataset: Use code above
- Integrate into application: Import `Pipeline2` class

## Support

See main README for architecture details and design decisions.
