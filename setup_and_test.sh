#!/bin/bash

# Setup and Test Script for Pipeline 2
# Installs dependencies and runs quick validation

echo "========================================="
echo "  Pipeline 2 Setup and Test"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "(This may take 5-10 minutes)"
echo ""

pip install -q -r requirements_pipeline2.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✓ Dependencies installed"

# Check for API key
echo ""
echo "Checking for GEMINI_API_KEY..."

if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  Warning: GEMINI_API_KEY not set"
    echo "   Set it with: export GEMINI_API_KEY='your-key-here'"
    echo "   Or create a .env file"
else
    echo "✓ GEMINI_API_KEY found"
fi

# Check for dataset
echo ""
echo "Checking for TAT-QA dataset..."

if [ -f "tatqa_dataset_test.json" ]; then
    echo "✓ Dataset found"
else
    echo "⚠️  Warning: tatqa_dataset_test.json not found"
    echo "   Download from TAT-QA benchmark and place in this directory"
fi

# Run quick module tests
echo ""
echo "Testing individual modules..."
echo ""

echo "[1/4] Testing RAG module..."
python3 -c "from rag_module import RAGRetriever; print('  ✓ RAG module OK')"

echo "[2/4] Testing Table Agent..."
python3 -c "from table_agent import TableAgent; print('  ✓ Table Agent OK')"

echo "[3/4] Testing Orchestrator..."
python3 -c "from orchestrator import GeminiOrchestrator; print('  ✓ Orchestrator OK')"

echo "[4/4] Testing Pipeline..."
python3 -c "from pipeline_v2 import Pipeline2; print('  ✓ Pipeline OK')"

echo ""
echo "========================================="
echo "  Setup Complete"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Set GEMINI_API_KEY: export GEMINI_API_KEY='your-key'"
echo "2. Place tatqa_dataset_test.json in this directory"
echo "3. Run demo: python3 demo.py"
echo ""
