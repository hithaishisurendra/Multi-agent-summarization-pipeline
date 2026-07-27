"""
RAG Module: BGE Embeddings + FAISS Vector Search
Retrieves top-K most relevant passages for a given question.
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss


class RAGRetriever:
    """
    Retrieval-Augmented Generation module using BGE embeddings and FAISS.
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        """
        Initialize RAG retriever with BGE model.

        Args:
            model_name: HuggingFace model for embeddings
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.passages = []
        self.metadata = []

    def embed_dataset(self, dataset_path: Path, max_examples: int = None):
        """
        Embed all paragraphs from TAT-QA dataset.

        Args:
            dataset_path: Path to TAT-QA JSON file
            max_examples: Limit number of examples (None = all)
        """
        print(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'r') as f:
            data = json.load(f)

        if max_examples:
            data = data[:max_examples]
            print(f"Limited to {max_examples} examples")

        all_passages = []
        all_metadata = []

        for idx, entry in enumerate(data):
            uid = entry.get("table", {}).get("uid", f"entry_{idx}")
            paragraphs = entry.get("paragraphs", [])

            # Sort paragraphs by order
            paragraphs = sorted(paragraphs, key=lambda x: x.get("order", 0))

            for para in paragraphs:
                text = para.get("text", "").strip()
                if text:
                    all_passages.append(text)
                    all_metadata.append({
                        "uid": uid,
                        "order": para.get("order", 0),
                        "entry_idx": idx
                    })

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(data)} entries")

        print(f"\nTotal passages to embed: {len(all_passages)}")

        # Embed all passages
        print("Generating embeddings...")
        embeddings = self.model.encode(
            all_passages,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        self.passages = all_passages
        self.metadata = all_metadata

        # Build FAISS index
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))

        print(f"FAISS index built with {self.index.ntotal} vectors")

    def retrieve_top_k(self, question: str, k: int = 3):
        """
        Retrieve top-K most relevant passages for a question.

        Args:
            question: User question
            k: Number of passages to retrieve

        Returns:
            List of dicts with passage, metadata, and similarity score
        """
        if self.index is None:
            raise ValueError("Index not built. Call embed_dataset() first.")

        # Embed question
        question_embedding = self.model.encode([question], convert_to_numpy=True)

        # Search FAISS index
        distances, indices = self.index.search(
            question_embedding.astype('float32'),
            k
        )

        # Format results
        results = []
        for idx, (passage_idx, distance) in enumerate(zip(indices[0], distances[0])):
            results.append({
                "rank": idx + 1,
                "passage": self.passages[passage_idx],
                "metadata": self.metadata[passage_idx],
                "similarity_score": float(1 / (1 + distance))  # Convert L2 distance to similarity
            })

        return results

    def save_index(self, index_path: Path, metadata_path: Path):
        """Save FAISS index and metadata for reuse."""
        print(f"Saving FAISS index to {index_path}")
        faiss.write_index(self.index, str(index_path))

        print(f"Saving metadata to {metadata_path}")
        with open(metadata_path, 'w') as f:
            json.dump({
                "passages": self.passages,
                "metadata": self.metadata
            }, f, indent=2)

    def load_index(self, index_path: Path, metadata_path: Path):
        """Load pre-built FAISS index and metadata."""
        print(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(str(index_path))

        print(f"Loading metadata from {metadata_path}")
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            self.passages = data["passages"]
            self.metadata = data["metadata"]

        print(f"Loaded index with {self.index.ntotal} vectors")


if __name__ == "__main__":
    """Quick test with sample data"""

    # Initialize retriever
    retriever = RAGRetriever()

    # Example: embed small dataset (change path to your actual file)
    # retriever.embed_dataset(Path("tatqa_dataset_test.json"), max_examples=10)

    # Example: retrieve passages
    # question = "What was the revenue growth in 2019?"
    # results = retriever.retrieve_top_k(question, k=3)
    #
    # print(f"\nTop-3 passages for: '{question}'")
    # for r in results:
    #     print(f"\nRank {r['rank']} (similarity: {r['similarity_score']:.3f})")
    #     print(f"Passage: {r['passage'][:200]}...")

    print("RAG module ready. Uncomment example code to test.")
