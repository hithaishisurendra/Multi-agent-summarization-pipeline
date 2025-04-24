# ===== CoT Prompts =====
TABUSYNTH_TEMPLATE = '''
Table Extraction (TabuSynth)

Table text:
{raw_table_data}

Goal: pull out each label, its number(s), and any years.
Please think out loud in numbered steps:
1. Spot the first field name.
2. Show how you locate its value.
3. Explain how you identify any year in parentheses.
4. Continue until every field is parsed.
'''

CONTEXTRON_TEMPLATE = '''
Context Filtering (Contextron)

Question:
{question}

Your role / background:
• Persona: {persona_label}
• Context note: {persona_context}

Raw table text:
{raw_table_data}

Step-by-step, explain how you choose which table facts answer the question for this role:
1. Restate what the question is asking.
2. Highlight matching terms in the table.
3. Bring in your background note—what matters most to this persona?
4. Select the relevant figures and say why.
'''

VISURA_TEMPLATE = '''
Visual Cues (Visura)

Raw table text:
{raw_table_data}

Please think out loud in numbered steps:
1. Note any separators (colons, semicolons).
2. Explain what parentheses (or other marks) imply.
3. Call out any other visual hints that guide interpretation.
'''

SUMMACRAFT_TEMPLATE = '''
Final Summary (SummaCraft)

Question:
{question}

TabuSynth notes:
{tabu_cot}

Contextron notes:
{contextron_cot}

Visura notes:
{visura_cot}

Persona for tone & focus:
{persona_label}

Using these three human-style reasoning traces, write a concise answer.
Begin with "Because..." to show your brief rationale, then deliver the final result.
'''

# ===== LLM interface =====
def send_llm(prompt: str) -> str:
    """ Sends `prompt` to an LLM and returns its string response (Chain-of-Thought or final summary).
    """
    pass

# ===== Agents =====
def run_tabu_synth(raw_table_data):
    prompt = TABUSYNTH_TEMPLATE.format(raw_table_data=raw_table_data)
    return send_llm(prompt)

def run_contextron(question, persona_label, persona_context, raw_table_data):
    prompt = CONTEXTRON_TEMPLATE.format(
        question=question,
        persona_label=persona_label,
        persona_context=persona_context,
        raw_table_data=raw_table_data,
    )
    return send_llm(prompt)

def run_visura(raw_table_data):
    prompt = VISURA_TEMPLATE.format(raw_table_data=raw_table_data)
    return send_llm(prompt)

def run_summa_craft(question, persona_label, tabu_cot, contextron_cot, visura_cot):
    prompt = SUMMACRAFT_TEMPLATE.format(
        question=question,
        persona_label=persona_label,
        tabu_cot=tabu_cot,
        contextron_cot=contextron_cot,
        visura_cot=visura_cot,
    )
    return send_llm(prompt)


def summarize_table(raw_table_data, question, persona_label, persona_context):
    tabu_cot = run_tabu_synth(raw_table_data)
    contextron_cot = run_contextron(question, persona_label, persona_context, raw_table_data)
    visura_cot = run_visura(raw_table_data)
    final_answer = run_summa_craft(
        question,
        persona_label,
        tabu_cot,
        contextron_cot,
        visura_cot,
    )
    return final_answer

# Example
if __name__ == '__main__':
    table = "Stock options: $2,756 (2019), $2,926 (2018); RSUs: 955 (2019), 1,129 (2018)"
    question = "What was the amount of unrecognized stock-based compensation expense related to unvested employee stock options in 2019?"
    persona_label = "Compensation Analyst"
    persona_context = "As part of our compensation cost projections, we need to evaluate unrecognized expenses in 2019 to understand how it will impact future financial statements."

    answer = summarize_table(table, question, persona_label, persona_context)
    print(answer)
