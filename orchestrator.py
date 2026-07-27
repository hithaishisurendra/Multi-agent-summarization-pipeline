"""
Orchestrator: Gemini-based synthesis of multi-agent outputs
Combines table facts, context, and persona into final summary.
"""

import os
import google.generativeai as genai


class GeminiOrchestrator:
    """
    Orchestrates multi-agent outputs using Gemini 2.5 Flash.
    Generates persona-adapted summaries based on user expertise level.
    """

    # Persona definitions for adaptation
    PERSONA_DESCRIPTIONS = {
        "financial analyst": {
            "description": "Expert who needs detailed segment-by-segment breakdowns with precise numbers and quantitative analysis",
            "tone": "Technical and detailed",
            "requirements": ["Specific metrics", "Segment breakdowns", "Year-over-year comparisons", "Detailed calculations"]
        },
        "business manager": {
            "description": "Professional who needs actionable insights, trends, and operational implications",
            "tone": "Strategic and action-oriented",
            "requirements": ["Key trends", "Business implications", "Actionable recommendations", "Clear next steps"]
        },
        "novice user": {
            "description": "Beginner who needs simple explanations without jargon or technical terminology",
            "tone": "Simple and accessible",
            "requirements": ["Plain language", "No jargon", "Simple explanations", "Clear context"]
        },
        "investor": {
            "description": "Stakeholder focused on ROI, strategic business implications, and financial performance",
            "tone": "Strategic and financially-focused",
            "requirements": ["ROI implications", "Growth indicators", "Strategic positioning", "Risk factors"]
        },
        "technical expert": {
            "description": "Specialist focused on methodology, data quality, and analytical rigor",
            "tone": "Methodological and precise",
            "requirements": ["Methodology details", "Data quality notes", "Calculation methodology", "Assumptions"]
        },
        "CFO": {
            "description": "C-suite executive who needs strategic implications, high-level insights, and concise summaries",
            "tone": "Strategic and concise",
            "requirements": ["Strategic implications", "High-level metrics", "Business impact", "Executive summary"]
        }
    }

    def __init__(self, api_key: str = None, model_name: str = "gemini-1.5-flash"):
        """
        Initialize Gemini orchestrator.

        Args:
            api_key: Gemini API key (if None, reads from env)
            model_name: Gemini model to use
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=model_name)
        print(f"Gemini orchestrator initialized with {model_name}")

    def build_orchestration_prompt(
        self,
        question: str,
        table_facts: dict,
        enriched_context: dict,
        persona: str,
        rag_passages: list = None
    ) -> str:
        """
        Build prompt for Gemini combining all agent outputs.

        Args:
            question: User question
            table_facts: Output from Table Agent
            enriched_context: Output from Context Agent
            persona: User persona from Selectra
            rag_passages: Retrieved passages from RAG (optional)

        Returns:
            Orchestration prompt
        """
        prompt_parts = []

        # Add persona context if available
        persona_context = ""
        if persona in self.PERSONA_DESCRIPTIONS:
            persona_info = self.PERSONA_DESCRIPTIONS[persona]
            persona_context = f"{persona_info['description']}"
            prompt_parts.append(f"You are generating a summary for a {persona} ({persona_context}).")
        else:
            prompt_parts.append(f"You are generating a summary for a {persona}.")

        prompt_parts.append(f"\nQuestion: {question}\n")

        # Add RAG passages if available
        if rag_passages:
            prompt_parts.append("Retrieved Relevant Passages:")
            for idx, passage in enumerate(rag_passages, 1):
                score = passage.get("similarity_score", 0)
                text = passage.get("passage", "")
                prompt_parts.append(f"{idx}. (Similarity: {score:.3f}) {text[:300]}...")

        # Add table facts
        prompt_parts.append("\n--- Table Agent Output ---")
        facts = table_facts.get("extracted_facts", [])
        if facts:
            prompt_parts.append("Extracted Facts:")
            for fact in facts:
                prompt_parts.append(f"  - {fact}")
        else:
            prompt_parts.append("No specific facts extracted from table")

        cot_table = table_facts.get("cot_trace", "")
        if cot_table:
            prompt_parts.append(f"\nTable Agent CoT Trace:\n{cot_table}")

        # Add enriched context
        prompt_parts.append("\n--- Context Agent Output ---")
        context_text = enriched_context.get("enriched_context", "")
        if context_text:
            prompt_parts.append(f"Enriched Context: {context_text}")
        else:
            prompt_parts.append("No additional context available")

        cot_context = enriched_context.get("cot_trace", "")
        if cot_context:
            prompt_parts.append(f"\nContext Agent CoT Trace:\n{cot_context}")

        # Add instructions with persona-specific requirements
        prompt_parts.append("\n--- Instructions ---")
        prompt_parts.append(
            "Using the information above, generate a concise answer adapted to the persona's expertise level."
        )

        # Add persona-specific requirements if available
        if persona in self.PERSONA_DESCRIPTIONS:
            requirements = self.PERSONA_DESCRIPTIONS[persona]["requirements"]
            prompt_parts.append(f"\nPersona requirements: {', '.join(requirements)}")

        prompt_parts.append(
            "\nUse chain-of-thought reasoning to show your work, then provide the final summary."
        )
        prompt_parts.append(
            "\nFormat:\n1. Chain-of-Thought (brief reasoning steps)\n2. Summary (final answer adapted to persona)"
        )

        return "\n".join(prompt_parts)

    def generate_summary(
        self,
        question: str,
        table_facts: dict,
        enriched_context: dict,
        persona: str,
        rag_passages: list = None
    ) -> dict:
        """
        Generate persona-adapted summary using Gemini.

        Args:
            question: User question
            table_facts: Output from Table Agent
            enriched_context: Output from Context Agent
            persona: User persona
            rag_passages: Retrieved RAG passages

        Returns:
            Dict with summary and reasoning
        """
        prompt = self.build_orchestration_prompt(
            question, table_facts, enriched_context, persona, rag_passages
        )

        try:
            response = self.model.generate_content(prompt)
            output_text = response.text.strip()

            # Parse output (simple split on common patterns)
            parts = output_text.split("Summary:", 1)
            if len(parts) == 2:
                cot = parts[0].strip()
                summary = parts[1].strip()
            else:
                # Fallback: treat entire response as summary
                cot = ""
                summary = output_text

            return {
                "summary": summary,
                "cot_reasoning": cot,
                "persona": persona,
                "raw_output": output_text
            }

        except Exception as e:
            print(f"Gemini API error: {e}")
            return {
                "summary": f"Error generating summary: {e}",
                "cot_reasoning": "",
                "persona": persona,
                "raw_output": ""
            }

    def synthesize_multi_agent(
        self,
        question: str,
        table_agent_output: dict,
        context_agent_output: dict,
        persona_agent_output: str,
        rag_passages: list = None
    ) -> dict:
        """
        Full orchestration: synthesize all agent outputs.

        Args:
            question: User question
            table_agent_output: Table Agent results
            context_agent_output: Context Agent results
            persona_agent_output: Persona classification
            rag_passages: RAG retrieved passages

        Returns:
            Final orchestrated summary with all metadata
        """
        summary_result = self.generate_summary(
            question=question,
            table_facts=table_agent_output,
            enriched_context=context_agent_output,
            persona=persona_agent_output,
            rag_passages=rag_passages
        )

        # Combine everything into full output
        return {
            "question": question,
            "persona": persona_agent_output,
            "rag_passages": rag_passages,
            "table_agent": table_agent_output,
            "context_agent": context_agent_output,
            "orchestrator": summary_result,
            "final_summary": summary_result["summary"]
        }


if __name__ == "__main__":
    """Quick test"""

    # Initialize (requires GEMINI_API_KEY in environment)
    try:
        orchestrator = GeminiOrchestrator()

        # Mock agent outputs
        question = "What was the revenue growth in 2019?"

        table_facts = {
            "extracted_facts": ["Revenue 2018: $1.2M", "Revenue 2019: $1.5M"],
            "cot_trace": "Step 1: Located Revenue column\nStep 2: Found 2018 value: $1.2M\nStep 3: Found 2019 value: $1.5M"
        }

        enriched_context = {
            "enriched_context": "Revenue growth driven by defense contracts in 2019.",
            "cot_trace": "Identified key growth drivers: defense sector expansion"
        }

        persona = "financial analyst"

        print(f"Question: {question}\n")
        print("Generating orchestrated summary...\n")

        result = orchestrator.generate_summary(
            question, table_facts, enriched_context, persona
        )

        print(f"CoT Reasoning:\n{result['cot_reasoning']}\n")
        print(f"Summary:\n{result['summary']}")

    except ValueError as e:
        print(f"Error: {e}")
        print("Set GEMINI_API_KEY environment variable to test orchestrator")

    print("\nOrchestrator ready.")
