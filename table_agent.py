"""
Table Agent: TAPAS-based structured fact extraction
Extracts facts from tables with Chain-of-Thought traces.
"""

import json
import pandas as pd
from pathlib import Path
from transformers import TapasTokenizer, TapasForQuestionAnswering
import torch


class TableAgent:
    """
    TAPAS-based agent for extracting structured facts from tables.
    """

    def __init__(self, model_name: str = "google/tapas-base-finetuned-wtq"):
        """
        Initialize TAPAS model for table QA.

        Args:
            model_name: HuggingFace TAPAS model
        """
        print(f"Loading TAPAS model: {model_name}")
        self.tokenizer = TapasTokenizer.from_pretrained(model_name)
        self.model = TapasForQuestionAnswering.from_pretrained(model_name)
        print("TAPAS model loaded successfully")

    def parse_tatqa_table(self, table_data: dict) -> pd.DataFrame:
        """
        Convert TAT-QA table format to pandas DataFrame.

        Args:
            table_data: TAT-QA table dict with 'table' key

        Returns:
            pandas DataFrame
        """
        table_list = table_data.get("table", [])
        if not table_list or len(table_list) < 2:
            return pd.DataFrame()

        # First row is header
        headers = table_list[0]
        # Remaining rows are data
        rows = table_list[1:]

        df = pd.DataFrame(rows, columns=headers)
        return df

    def extract_facts(self, table_df: pd.DataFrame, question: str):
        """
        Extract facts from table using TAPAS.

        Args:
            table_df: pandas DataFrame
            question: User question

        Returns:
            Dict with extracted facts and CoT trace
        """
        if table_df.empty:
            return {
                "extracted_facts": [],
                "cot_trace": "Table is empty or malformed",
                "cells": [],
                "aggregation": None
            }

        # Prepare inputs for TAPAS
        inputs = self.tokenizer(
            table=table_df,
            queries=question,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get predicted cells
        predicted_cells = self.tokenizer.convert_logits_to_predictions(
            inputs,
            outputs.logits.detach(),
            outputs.logits_aggregation.detach() if hasattr(outputs, 'logits_aggregation') else None
        )

        # Extract cell coordinates and values
        cell_coords = predicted_cells[0]
        extracted_facts = []
        cells_info = []

        for coords in cell_coords:
            row_idx, col_idx = coords
            if row_idx < len(table_df) and col_idx < len(table_df.columns):
                col_name = table_df.columns[col_idx]
                cell_value = table_df.iloc[row_idx, col_idx]
                fact = f"{col_name}: {cell_value}"
                extracted_facts.append(fact)
                cells_info.append({
                    "row": int(row_idx),
                    "col": int(col_idx),
                    "column_name": col_name,
                    "value": str(cell_value)
                })

        # Generate CoT trace
        cot_trace = self._generate_cot_trace(table_df, question, cells_info)

        # Get aggregation if available
        aggregation = None
        if hasattr(outputs, 'logits_aggregation'):
            agg_logits = outputs.logits_aggregation.detach()
            aggregation = torch.argmax(agg_logits, dim=-1).item()

        return {
            "extracted_facts": extracted_facts,
            "cot_trace": cot_trace,
            "cells": cells_info,
            "aggregation": aggregation
        }

    def _generate_cot_trace(self, table_df: pd.DataFrame, question: str, cells_info: list) -> str:
        """
        Generate Chain-of-Thought trace for extraction process.

        Args:
            table_df: Table DataFrame
            question: User question
            cells_info: List of extracted cell information

        Returns:
            CoT trace string
        """
        trace_steps = []

        trace_steps.append(f"Step 1: Analyzed question: '{question}'")
        trace_steps.append(f"Step 2: Table has {len(table_df)} rows and {len(table_df.columns)} columns")
        trace_steps.append(f"Step 3: Column names: {', '.join(table_df.columns.tolist())}")

        if cells_info:
            trace_steps.append(f"Step 4: Located {len(cells_info)} relevant cells:")
            for idx, cell in enumerate(cells_info, 1):
                trace_steps.append(
                    f"  {idx}. Row {cell['row']}, Column '{cell['column_name']}': {cell['value']}"
                )
        else:
            trace_steps.append("Step 4: No specific cells matched the question")

        trace_steps.append("Step 5: Extracted facts listed above")

        return "\n".join(trace_steps)

    def process_tatqa_entry(self, entry: dict, question: str):
        """
        Process a full TAT-QA entry (convenience method).

        Args:
            entry: TAT-QA JSON entry
            question: User question

        Returns:
            Dict with extracted facts and CoT trace
        """
        table_data = entry.get("table", {})
        table_df = self.parse_tatqa_table(table_data)

        return self.extract_facts(table_df, question)


if __name__ == "__main__":
    """Quick test"""

    # Initialize agent
    agent = TableAgent()

    # Example table
    sample_table = pd.DataFrame({
        "Year": ["2018", "2019", "2020"],
        "Revenue": ["1.2M", "1.5M", "1.8M"],
        "Profit": ["0.3M", "0.4M", "0.5M"]
    })

    question = "What was the revenue in 2019?"

    print(f"\nQuestion: {question}")
    print(f"\nTable:\n{sample_table}\n")

    result = agent.extract_facts(sample_table, question)

    print("Extracted Facts:")
    for fact in result["extracted_facts"]:
        print(f"  - {fact}")

    print(f"\nCoT Trace:\n{result['cot_trace']}")

    print("\nTable Agent ready.")
