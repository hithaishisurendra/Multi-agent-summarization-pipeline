"""
Pipeline 2: Sequential Multi-Agent with RAG
Complete end-to-end flow: RAG → TAPAS → Context → Selectra → Orchestrator
"""

import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from rag_module import RAGRetriever
from table_agent import TableAgent
from orchestrator import GeminiOrchestrator


class Pipeline2:
    """
    Full Pipeline 2 implementation with RAG-augmented multi-agent architecture.
    """

    def __init__(
        self,
        gemini_api_key: str = None,
        rag_model: str = "BAAI/bge-large-en-v1.5",
        table_model: str = "google/tapas-base-finetuned-wtq",
        context_model: str = "google/flan-t5-xl",
        persona_model: str = "google/flan-t5-small"
    ):
        """
        Initialize all agents in Pipeline 2.

        Args:
            gemini_api_key: Gemini API key for orchestrator
            rag_model: BGE embedding model
            table_model: TAPAS model for table extraction
            context_model: FLAN-T5 model for context enrichment
            persona_model: FLAN-T5 model for persona inference
        """
        print("Initializing Pipeline 2...")

        # RAG Retriever
        print("\n[1/5] Loading RAG Retriever...")
        self.rag = RAGRetriever(model_name=rag_model)

        # Table Agent
        print("\n[2/5] Loading Table Agent...")
        self.table_agent = TableAgent(model_name=table_model)

        # Context Agent (FLAN-T5-XL)
        print("\n[3/5] Loading Context Agent...")
        self.context_tokenizer = AutoTokenizer.from_pretrained(context_model)
        self.context_model = AutoModelForSeq2SeqLM.from_pretrained(context_model)

        # Persona Agent (FLAN-T5-Small)
        print("\n[4/5] Loading Persona Agent...")
        self.persona_tokenizer = AutoTokenizer.from_pretrained(persona_model)
        self.persona_model = AutoModelForSeq2SeqLM.from_pretrained(persona_model)

        # Orchestrator
        print("\n[5/5] Loading Gemini Orchestrator...")
        self.orchestrator = GeminiOrchestrator(api_key=gemini_api_key)

        print("\n✓ Pipeline 2 initialized successfully\n")

    def run_context_agent(self, rag_passages: list) -> dict:
        """
        Process RAG-retrieved passages with Context Agent.

        Args:
            rag_passages: Top-K passages from RAG

        Returns:
            Dict with enriched context and CoT trace
        """
        # Combine passages
        passages_text = " ".join([p["passage"] for p in rag_passages])

        # Generate prompt
        prompt = f"Extract and condense contextual metadata: {passages_text}"

        # Tokenize and generate
        inputs = self.context_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        outputs = self.context_model.generate(
            **inputs,
            max_new_tokens=150
        )
        enriched = self.context_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Simple CoT trace
        cot_trace = f"Processed {len(rag_passages)} retrieved passages\n"
        cot_trace += f"Combined {len(passages_text)} characters of context\n"
        cot_trace += "Condensed into enriched summary"

        return {
            "enriched_context": enriched,
            "cot_trace": cot_trace,
            "num_passages": len(rag_passages)
        }

    def run_persona_agent(self, question: str) -> str:
        """
        Infer user persona from question phrasing.

        Args:
            question: User question

        Returns:
            Persona string
        """
        prompt = (
            f"Classify the user who asked this financial question:\n"
            f"Question: \"{question}\"\n"
            f"User type (choose one): financial analyst, business manager, "
            f"technical expert, novice user, investor.\nAnswer:"
        )

        inputs = self.persona_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        )
        outputs = self.persona_model.generate(**inputs, max_new_tokens=10)
        persona = self.persona_tokenizer.decode(outputs[0], skip_special_tokens=True)

        return persona.strip()

    def process_single_question(
        self,
        entry: dict,
        question: str,
        use_rag: bool = True,
        top_k: int = 3
    ) -> dict:
        """
        Process a single question through full Pipeline 2.

        Args:
            entry: TAT-QA entry dict
            question: User question
            use_rag: Whether to use RAG retrieval (False = use all paragraphs)
            top_k: Number of passages to retrieve

        Returns:
            Complete pipeline output with all agent results
        """
        print(f"\n{'='*60}")
        print(f"Processing Question: {question}")
        print(f"{'='*60}\n")

        # Step 1: RAG Retrieval
        rag_passages = None
        if use_rag and self.rag.index is not None:
            print("[Step 1/5] RAG Retrieval...")
            rag_passages = self.rag.retrieve_top_k(question, k=top_k)
            print(f"  Retrieved {len(rag_passages)} passages")
        else:
            print("[Step 1/5] RAG Retrieval... SKIPPED (using all paragraphs)")

        # Step 2: Table Agent
        print("\n[Step 2/5] Table Agent Extraction...")
        table_output = self.table_agent.process_tatqa_entry(entry, question)
        print(f"  Extracted {len(table_output['extracted_facts'])} facts")

        # Step 3: Context Agent
        print("\n[Step 3/5] Context Agent Processing...")
        if rag_passages:
            context_output = self.run_context_agent(rag_passages)
        else:
            # Fallback: use all paragraphs (old behavior)
            paragraphs = entry.get("paragraphs", [])
            paragraphs = sorted(paragraphs, key=lambda x: x.get("order", 0))
            all_text = " ".join([p["text"] for p in paragraphs])
            context_output = self.run_context_agent([{"passage": all_text}])
        print(f"  Generated enriched context")

        # Step 4: Persona Agent
        print("\n[Step 4/5] Persona Inference...")
        persona = self.run_persona_agent(question)
        print(f"  Detected persona: {persona}")

        # Step 5: Orchestrator
        print("\n[Step 5/5] Gemini Orchestration...")
        final_output = self.orchestrator.synthesize_multi_agent(
            question=question,
            table_agent_output=table_output,
            context_agent_output=context_output,
            persona_agent_output=persona,
            rag_passages=rag_passages
        )
        print(f"  Generated final summary")

        print(f"\n{'='*60}")
        print("Pipeline Complete")
        print(f"{'='*60}\n")

        return final_output

    def build_rag_index(self, dataset_path: Path, max_examples: int = None):
        """
        Build RAG index from dataset.

        Args:
            dataset_path: Path to TAT-QA JSON
            max_examples: Limit number of examples
        """
        print("\nBuilding RAG index...")
        self.rag.embed_dataset(dataset_path, max_examples=max_examples)
        print("RAG index ready\n")


if __name__ == "__main__":
    """Quick test"""

    import os

    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: Set GEMINI_API_KEY environment variable")
        print("Example: export GEMINI_API_KEY='your-key-here'")
        exit(1)

    print("Pipeline 2 Test Script")
    print("=" * 60)

    # Initialize pipeline (this will take a few minutes)
    pipeline = Pipeline2()

    # To test with actual data:
    # 1. Build RAG index
    #    pipeline.build_rag_index(Path("tatqa_dataset_test.json"), max_examples=10)
    #
    # 2. Load a TAT-QA entry
    #    with open("tatqa_dataset_test.json", "r") as f:
    #        data = json.load(f)
    #    entry = data[0]
    #
    # 3. Process a question
    #    question = "What was the revenue in 2019?"
    #    result = pipeline.process_single_question(entry, question)
    #
    # 4. Print results
    #    print(json.dumps(result, indent=2))

    print("\nPipeline 2 ready. Uncomment test code to run with real data.")
