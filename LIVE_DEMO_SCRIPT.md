# Live Demo Script: Code Walkthrough Guide

**Total Time**: 10-15 minutes with natural interruptions
**Format**: Interleaved explanation → code → pause for questions

---

## Opening: Problem + Why (2 minutes, no code yet)

**YOU SAY** (30 seconds):
> "I built a multi-agent QA system for financial reports that mix tables and text. The core problem: a single LLM can't handle this well—they either focus too much on tables and miss context, or read text but fail at arithmetic. Plus, 47 paragraphs per report exceeds token limits.
>
> So I designed a pipeline where RAG cuts 7,500 tokens to 450, then specialized agents handle table extraction, context, and persona inference. Gemini orchestrates everything. It costs $0.000275 per query—160 times cheaper than GPT-4—and runs in 1.7 seconds."

**PAUSE FOR QUESTIONS** ← Interviewer will likely ask "how does it work?" or "walk me through it"

---

## Architecture Overview (1 minute, show diagram)

**YOU SAY** (30 seconds):
> "Let me show you the architecture. Five stages: RAG retrieval, table agent, context agent, persona inference, orchestrator. Each passes outputs AND reasoning traces forward—that's key. Unlike router-based approaches that silo information, this preserves full context."

**SHOW**: `pipeline 2.png` diagram on screen

**YOU SAY** (15 seconds):
> "Now let me walk you through the actual code, following the execution path."

**PAUSE** ← They'll say "sounds good" or ask about a specific component

---

## Code Walkthrough: Follow the Execution Path

### File 1: RAG Module - The Entry Point (2 minutes)

**OPEN**: `rag_module.py`

**YOU SAY** (20 seconds):
> "RAG is the entry point. When a user asks 'What was revenue in 2019?', I embed that question with BGE and use FAISS to find the 3 most semantically similar paragraphs. This is how I cut 7,500 tokens to 450."

**SCROLL TO** lines 91-111 (`retrieve_top_k` function):

**YOU SAY** (30 seconds while pointing):
> "Here's the core retrieval logic. Line 106: embed the question with BGE. Line 109: FAISS searches the pre-built index. Line 121: I convert L2 distance to similarity score—this is what tells downstream agents which passages are most relevant."

**SHOW CODE**:
```python
# Line 106-112 (point at screen)
question_embedding = self.model.encode([question], convert_to_numpy=True)

distances, indices = self.index.search(
    question_embedding.astype('float32'),
    k
)
```

**YOU SAY** (15 seconds):
> "The result: top-3 passages with similarity scores like 0.89, 0.85, 0.81. This takes 55 milliseconds—50ms for encoding, 5ms for FAISS search."

**PAUSE FOR QUESTIONS** ← Natural breakpoint, they might ask "why FAISS?" or "why K=3?"

**IF ASKED "Why K=3?"**:
> "I tested K=1, 3, 5, 10. K=1 gave 62% coverage—too narrow. K=5 gave 91% but introduced noise that confused downstream agents. K=3 hits 87% coverage with minimal noise—that's the sweet spot."

---

### File 2: Table Agent - Structured Extraction (2 minutes)

**OPEN**: `table_agent.py`

**YOU SAY** (20 seconds):
> "Next, the table agent. TAPAS is pre-trained on table QA with cell-level attention. Unlike GPT-4 which treats tables as text, TAPAS predicts cell coordinates directly."

**SCROLL TO** lines 72-88 (`extract_facts` function):

**YOU SAY** (30 seconds while pointing):
> "Line 72: TAPAS tokenizer converts the DataFrame to special table tokens. Line 82: model inference. Line 85: `convert_logits_to_predictions` gives me actual cell coordinates—row, column—not just text spans."

**SCROLL TO** lines 96-108 (show the output structure):

**YOU SAY** (20 seconds):
> "The output is structured: extracted facts like 'Revenue 2018: $1.2M', plus a chain-of-thought trace showing which cells it looked at. This transparency is critical—if it extracts the wrong cell, I can debug it."

**PAUSE FOR QUESTIONS** ← They might ask "what's the F1 score?" or "why not GPT-4?"

**IF ASKED "Why not GPT-4 for tables?"**:
> "GPT-4 has no structured pre-training. It treats tables as markdown text and hallucinates cell values. TAPAS is purpose-built for this with cell-level attention—it can say 'row 2, column 3' directly."

---

### File 3: Orchestrator - The Prompt Engineering (3 minutes)

**OPEN**: `orchestrator.py`

**YOU SAY** (30 seconds):
> "Now the most interesting part—orchestration. This is where all the prompt engineering lives. Gemini receives all agent outputs, CoT traces, and persona, then generates an adapted summary."

**SCROLL TO** lines 17-48 (PERSONA_DESCRIPTIONS):

**YOU SAY** (30 seconds while scrolling):
> "First, I define explicit persona descriptions. Financial analyst needs detailed breakdowns, novice user needs simple language, CFO needs strategic focus. These get injected into the prompt to prime Gemini's generation."

**SCROLL TO** lines 52-103 (`build_orchestration_prompt` function):

**YOU SAY** (45 seconds, point at specific lines):
> "Here's the prompt structure. Line 94: persona framing comes FIRST—has to be at the start to prime generation. Lines 58-64: RAG passages with similarity scores—tells Gemini which passages are most relevant. Lines 66-90: section delimiters keep table facts separate from context. And critically, lines 77 and 89: CoT traces from each agent—Gemini sees HOW they arrived at answers."

**SHOW THE CRITICAL PROMPT SECTION** (lines 92-101):
```python
# Point at these lines
prompt_parts.append("You are generating a summary for a {persona}.")
# ...
prompt_parts.append("--- Table Agent Output ---")
# ...
prompt_parts.append("\nTable Agent CoT Trace:\n{cot_trace}")
```

**YOU SAY** (20 seconds):
> "The format instruction at the end—'1. Chain-of-Thought, 2. Summary'—makes the output parseable. I split on 'Summary:' to extract the reasoning separately."

**PAUSE FOR QUESTIONS** ← Natural breakpoint, they might ask about personas or prompt engineering

**IF ASKED "Show me how personas differ"**:
**SCROLL TO** README.md, lines 233-247 (Persona Adaptation Examples)

**YOU SAY** (30 seconds while showing):
> "Here's the same financial data adapted for different users. Financial analyst gets detailed breakdowns with segment analysis. Novice investor gets simple language—'mainly because of new defense contracts.' CFO gets strategic focus with action items. The system infers this from question phrasing alone."

---

### File 4: Pipeline - Putting It Together (2 minutes)

**OPEN**: `pipeline_v2.py`

**YOU SAY** (20 seconds):
> "Pipeline V2 ties everything together. The `process_single_question` method orchestrates all five stages sequentially."

**SCROLL TO** lines 131-200 (`process_single_question` function):

**YOU SAY** (45 seconds while scrolling slowly):
> "Here's the flow. Line 157: RAG retrieval. Line 165: Table agent gets table plus question. Line 170: Context agent receives ONLY the top-3 RAG passages—not all 47 paragraphs. That's the key—it's semantic filtering, not random sampling. Line 182: Persona inference. Line 187: Gemini orchestration receives everything."

**SCROLL TO** lines 64-102 (`run_context_agent` function):

**YOU SAY** (30 seconds):
> "The Context Agent prompt—line 78: 'Extract and condense contextual metadata.' I tested different phrasings. 'Summarize' was too verbose—250+ tokens. 'Extract facts' lost narrative context. This specific wording gives me 200 tokens with both facts AND why/how context."

**PAUSE FOR QUESTIONS** ← They might ask "why sequential not parallel?"

**IF ASKED "Why not parallelize?"**:
> "Table and Context agents both get input from RAG independently, so I could parallelize them. Current: 300ms + 800ms = 1.1s sequential. Parallel: max(300, 800) = 800ms. I'd save 300ms. I haven't done it yet for code simplicity—sequential is easier to debug. But in production, I'd definitely parallelize."

---

## End-to-End Example (2 minutes, show output)

**OPEN**: `outputs/demo_example_1.json` (if you have it, or show on screen)

**YOU SAY** (30 seconds):
> "Let me show you a complete example. Question: 'What was the percentage change in revenue from 2018 to 2019?'"

**SCROLL THROUGH THE JSON** (or narrate if showing):

**YOU SAY** (1 minute):
> "RAG retrieved three passages—similarity scores 0.89, 0.85, 0.81. Table agent extracted '2018 Total: $1.2M, 2019 Total: $1.5M'—its CoT trace shows 'Located Total column, found 2018 row.' Context agent condensed those passages into 'Revenue growth driven by Fixed Price defense contracts.' Persona inference: financial analyst. Gemini's final output: '25% increase from $1.2M to $1.5M, primarily driven by Fixed Price defense sector engagements.' Notice how it synthesizes both the table numbers AND the context about why."

**PAUSE FOR QUESTIONS** ← Natural ending point

---

## Transitions & Natural Pause Points

### When to Pause for Questions

**After each file (natural breakpoints)**:
1. After RAG retrieval explanation → "Any questions on the retrieval?"
2. After Table agent → "Before I move to context processing..."
3. After Orchestrator prompts → "Questions on the prompt engineering?"
4. After Pipeline flow → "That's the full pipeline—questions?"

### Transition Phrases (Use These)

**Moving between files**:
- "Now let me show you how this gets used in..."
- "The next piece is [component]—let me pull up that code..."
- "To see how this flows into the next stage..."

**Inviting questions**:
- "Does this make sense so far?"
- "Any questions before I move to [next component]?"
- "I can dig deeper into [X] if you want..."

**Handling interruptions gracefully**:
- "Great question—let me show you that in the code..."
- "Actually, that's exactly what I'm about to show you..."
- "I can come back to [X] after showing [Y], or we can dive in now—your call."

---

## File Order Quick Reference

**Execution path order** (recommended):
1. `rag_module.py` (55ms) - Entry point, show retrieval
2. `table_agent.py` (300ms) - Show TAPAS extraction
3. `orchestrator.py` (500ms) - Show prompt engineering + personas
4. `pipeline_v2.py` - Show how it all connects
5. `README.md` - Persona examples, tradeoffs
6. `outputs/demo_example_1.json` - End-to-end result

**Key sections by file**:

| File | Lines to Show | What to Say |
|------|--------------|-------------|
| `rag_module.py` | 91-111, 116-123 | "Embedding + FAISS search, 55ms total" |
| `table_agent.py` | 72-88, 96-108 | "TAPAS cell-level attention, CoT traces" |
| `orchestrator.py` | 17-48, 52-103, 135-149 | "Persona definitions, prompt structure, parsing" |
| `pipeline_v2.py` | 131-200, 64-102 | "Sequential flow, context agent prompt" |
| `README.md` | 233-247, 330-337 | "Persona examples, cost breakdown" |

---

## If Running Short on Time (Fast Version)

**5-minute compressed walkthrough**:

1. **Problem** (30 sec): "7,500 tokens, need specialized agents"
2. **Architecture diagram** (30 sec): "Five stages, context preservation"
3. **RAG code** (1 min): Show lines 106-112, "55ms, 94% token reduction"
4. **Orchestrator prompts** (2 min): Show lines 52-103, "Persona framing + CoT traces"
5. **End-to-end output** (1 min): Show JSON, "Table facts + context synthesis"

Skip: table_agent.py (mention "TAPAS with cell-level attention"), pipeline_v2.py (mention "sequential flow")

---

## Anticipating Follow-Up Questions

### Common Questions & Where to Show Code

**Q: "Why did Pipeline 1 fail?"**
→ Open README.md lines 464-481, explain hard routing

**Q: "How do you handle conflicts between agents?"**
→ Show orchestrator.py lines 135-149 (parsing logic), explain: "Gemini sees all CoT traces, can weight by confidence"

**Q: "What if RAG retrieves wrong passages?"**
→ Explain: "Happens 13% of time. Orchestrator detects low-quality context—vague statements, no specifics—and relies more on table facts. Ideal: add confidence scores."

**Q: "Show me the actual prompts"**
→ orchestrator.py lines 52-103 (already open), pipeline_v2.py line 78 for context agent

**Q: "How long does this take to run?"**
→ README.md lines 330-337 (latency breakdown table)

**Q: "What would you do differently?"**
→ README.md lines 667-699 (Future Improvements), or verbal: "Cross-encoder re-ranking, TabLLM, LLM-as-judge"

---

## Tips for Smooth Demo

### Before You Start
- Have all files open in tabs (don't waste time navigating)
- Zoom editor font to 14-16pt (readable on screen share)
- Close unrelated files
- Have README.md open for quick reference

### During Demo
- **Point at code with cursor** as you talk
- **Read key lines aloud**: "Line 106: I embed the question with BGE"
- **Pause for 2 seconds** after showing a code block (let them absorb)
- **Don't scroll too fast** (they need to read)
- **If they interrupt**: Stop immediately, answer, then "Should I continue or go deeper here?"

### Pacing
- **30-45 seconds per code section** (shorter than you think)
- **Pause every 1-2 minutes** for questions
- **If no questions**: "I'll keep going—stop me anytime"

### Energy
- **Vary your tone**: Excited about clever parts (prompt engineering), matter-of-fact about standard parts (imports)
- **Highlight trade-offs**: "I chose X over Y because..."
- **Show confidence**: "This is the part I'm proud of..." or "This was the hardest part..."

---

## Emergency Shortcuts

### If They're Bored (Skip Ahead)
"I can show you the interesting part—the prompt engineering where personas get adapted. Want to jump there?"

### If They're Lost (Zoom Out)
"Let me step back—here's the architecture diagram. We're at this stage..." [point]

### If Running Over Time
"I have 5 more minutes—should I show you [specific thing] or answer questions on what we've covered?"

### If They Want More Detail
"Want me to explain [X] in depth, or should I finish the overview first and come back?"

---

**Total realistic time**: 10-12 minutes with natural interruptions, 15 minutes if deep questions

**Key success metric**: Can they understand the flow without you monologuing for >2 minutes straight
