from transformers import pipeline

try:
    summarizer = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.1",
        device=0,           
    )
    print("Using Mistral-7B-Instruct")
except Exception:
    summarizer = pipeline(
        "text-generation",
        model="tiiuae/falcon-7b-instruct",
        device=0,
        torch_dtype="auto",
    )
    print("Falling back to Falcon-7B-Instruct")

personas = [
    "novice investor",
    "financial analyst",
    "executive summary for a CFO" 
]

def build_cot_prompt(df, context, extracted, persona):
    table_snippet = df.head().to_markdown() 
    return f"""
You are a {persona}.  Use chain‐of‐thought briefly to reason, then produce a concise summary.

Context:
{context}

Table (first rows):
{table_snippet}

Extracted facts:
{extracted}

What are the key takeaways?
"""

import json, pandas as pd

GOLD_JSON = "/content/drive/MyDrive/tatqa_dataset_test_gold.json"
gold = json.load(open(GOLD_JSON))[:50]

tables = {
    entry["table"]["uid"]: pd.DataFrame(
        entry["table"]["table"][1:],
        columns=entry["table"]["table"][0]
    )
    for entry in gold
}

import json

ctx = { r["uid"]: r["enriched_context"]
        for r in json.load(open("//content/drive/MyDrive/contexttron_50.json")) }

ts  = json.load(open("/content/drive/MyDrive/tabusynth_50.json"))


from transformers import pipeline

summarizer = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    device=0 
)

# Quick smoke test
test = summarizer("Summarize: The company’s revenue grew by 10% this quarter.", max_new_tokens=50)
print("Summarizer smoke test:", test[0]["generated_text"])


results = []

for rec in ts:
    uid       = rec["uid"]
    df        = tables[uid]
    context   = ctx.get(uid, "")
    extracted = rec["extracted"]

    for persona in personas:
        print(f"\n▶️  Generating for UID={uid}, persona={persona}")
        prompt = build_cot_prompt(df, context, extracted, persona) + "\n\nSUMMARY:\n"

        print("---- Prompt start ----")
        print(prompt)
        print("---- Prompt end ----\n")

        raw = summarizer(prompt, max_new_tokens=100)
        print("🤖 Raw pipeline output:", raw)

        out = raw[0].get("generated_text", "").strip()
        print(f"Done UID={uid} | {persona} SUMMARY:\n{out}\n")

        results.append({
            "uid": uid,
            "persona": persona,
            "summary": out
        })

import json
with open("summaries.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"All done — wrote {len(results)} summaries to summaries.json")