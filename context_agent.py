
import json
from pathlib import Path
from transformers import pipeline

class ContextronAgent:
    """
    Contextron: merges metadata/textual context and generates enriched summaries for tables.
    """
    def __init__(self,
                 model_name: str = "google/flan-t5-xl",
                 device: int = -1):
        print(f"Initializing ContextronAgent with model={model_name} on device={device}")
        self.pipe = pipeline(
            "text2text-generation",
            model=model_name,
            device=device,
            truncation=True,
            max_length=256
        )
        print("Model loaded successfully.")

    def build_context(self, paragraphs) -> str:
        texts = [
            p["text"]
            for p in sorted(paragraphs, key=lambda x: x["order"])
        ]
        return " ".join(texts)

    def enrich_context(self, context: str, instruction: str = "Summarize the key points:") -> str:
        prompt = f"{instruction} {context}"
        return self.pipe(prompt)[0]["generated_text"].strip()

    def process_dataset(self, dataset_path: Path, output_path: Path = None, resume: bool = False):
        print(f"Loading dataset from {dataset_path}")
        data = json.loads(dataset_path.read_text())
        processed = set()
        enriched_map = {}

        if resume and output_path and output_path.exists():
            print(f"Resuming from existing output: {output_path}")
            existing = json.loads(output_path.read_text())
            for entry in existing:
                uid = entry.get("table", {}).get("uid")
                if entry.get("enriched_context"):
                    processed.add(uid)
                    enriched_map[uid] = entry
            print(f"Found {len(processed)} already-processed entries; will skip those.")

        total = len(data)
        print(f"Total entries to consider: {total}")

        for idx, entry in enumerate(data, start=1):
            uid = entry.get("table", {}).get("uid")
            if uid in processed:
                print(f"[{idx}/{total}] Skipping already-processed UID: {uid}")
                continue

            print(f"[{idx}/{total}] Processing table UID: {uid}")
            paragraphs = entry.get("paragraphs", [])
            print(f"  Building context from {len(paragraphs)} paragraphs...")
            context = self.build_context(paragraphs)
            print(f"  Context length: {len(context)} characters")

            print("  Generating enriched context via model...")
            enriched = self.enrich_context(
                context,
                instruction="Extract and condense contextual metadata:"
            )
            print(f"  Enriched context length: {len(enriched)} characters")
            entry["enriched_context"] = enriched
            enriched_map[uid] = entry
            processed.add(uid)
            print(f"--- Entry {idx} complete ---\n")

            if idx % 100 == 0 and output_path:
                merged = [
                    enriched_map.get(e.get("table", {}).get("uid"), e)
                    for e in data
                ]
                output_path.write_text(
                    json.dumps(merged, indent=2, ensure_ascii=False)
                )
                print(f"Checkpoint: wrote {idx} entries to {output_path}")

        if output_path:
            print(f"Writing final enriched dataset to {output_path}")
            merged = [
                enriched_map.get(e.get("table", {}).get("uid"), e)
                for e in data
            ]
            output_path.write_text(
                json.dumps(merged, indent=2, ensure_ascii=False)
            )
            print("Enriched dataset written successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Contextron Agent on TAT-QA dataset with resume capability"
    )
    parser.add_argument("--input",   type=str, required=True,
                        help="Path to tatqa_dataset_test.json")
    parser.add_argument("--output",  type=str,
                        help="Path to save enriched JSON")
    parser.add_argument("--model",   type=str, default="google/flan-t5-xl",
                        help="HuggingFace model name")
    parser.add_argument("--device",  type=int, default=-1,
                        help="CUDA device; -1 for CPU")
    parser.add_argument("--resume",  action="store_true",
                        help="Resume from existing output file")
    args = parser.parse_args()

    agent = ContextronAgent(model_name=args.model, device=args.device)
    agent.process_dataset(
        dataset_path=Path(args.input),
        output_path=Path(args.output) if args.output else None,
        resume=args.resume
    )

