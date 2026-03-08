# NuritionVQA

Libraries :-
numpy
pillow
pydantic>=2
tqdm
datasets
transformers
accelerate
peft
evaluate
rapidfuzz
fastapi
uvicorn[standard]
python-multipart
streamlit
# NutritionVQA RAG - Core Dependencies
# LangChain & RAG
langchain>=0.1.0
langchain-core>=0.1.0
langchain-openai>=0.0.5
langchain-community>=0.0.20

# OpenAI
openai>=1.0.0

# Vector Store
chromadb>=0.4.22

# Dataset
datasets>=2.16.0
huggingface_hub>=0.20.0

# Image Processing & OCR
Pillow>=10.0.0
pytesseract>=0.3.10
python-dotenv>=1.0.0


# Implementation

# NutritionVQA RAG - Retrieval Augmented Generation

RAG implementation for the **NutritionVQA** multimodal platform using **LangChain** and **OpenAI**. Based on the [CoSyn paper](https://arxiv.org/abs/2502.14846) and [NutritionQA dataset](https://huggingface.co/datasets/yyupenn/NutritionQA).

## Architecture

```
User: image + question
         │
         ▼
    [Retriever] ← ChromaDB (NutritionQA Q&A embeddings)
         │
         ▼
    Retrieved context (similar Q&A pairs)
         │
         ▼
    [GPT-4V / GPT-4o-mini] ← image + question + context
         │
         ▼
    Answer
```

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set OpenAI API key**
   ```bash
   cp .env.example .env
   # Edit .env and add: OPENAI_API_KEY=sk-your-key
   ```

3. **Run (first run builds index automatically)**
   ```bash
   python run_rag.py
   ```

## Usage

### Demo (uses first NutritionQA example)
```bash
python run_rag.py
```

### Custom question and image
```bash
python run_rag.py --question "What's the amount of sodium per serving?" --image_path path/to/nutrition_label.jpg
```

### Rebuild index
```bash
python run_rag.py --build_index
python scripts/build_rag_index.py   # or build only
```

### Programmatic
```python
from rag import NutritionVQARAG, load_nutrition_qa

rag = NutritionVQARAG(persist_directory="chroma_nutrition_vqa", k=4)
dataset = load_nutrition_qa()
row = dataset[0]
answer = rag.query(question=row["descriptive_q"], image=row["image"])
print(answer)
```

## Dataset (NutritionQA)

- **Source:** [yyupenn/NutritionQA](https://huggingface.co/datasets/yyupenn/NutritionQA)
- **Size:** 50 examples (test split)
- **Fields:** `id`, `image`, `descriptive_q`, `descriptive_a`, `reasoning_q`, `reasoning_a`
- **Task:** Visual question answering on nutrition label photos

## RAG Pipeline (step by step)

1. **Load NutritionQA** → HuggingFace `datasets`
2. **Create documents** → 2 per example: descriptive Q&A + reasoning Q&A
3. **Build vector store** → ChromaDB + OpenAI `text-embedding-3-small`
4. **Retrieve** → Top-k similar Q&A by question embedding
5. **Generate** → GPT-4V with image + question + retrieved context

## Implementation Details (from rtrp.pdf p.14)

- Prompt structure: Topic Generation → Data Generation → Code Generation → Instruction Generation
- Question types: information retrieval (descriptive) + reasoning (multi-hop)
- CoT: Chain-of-thought improves NutritionQA (Table 7 in rtrp.pdf)

## Files

| File | Role |
|------|------|
| `rag/data_loader.py` | Load NutritionQA from HuggingFace |
| `rag/document_store.py` | Create LangChain documents for indexing |
| `rag/vector_store.py` | ChromaDB + OpenAI embeddings |
| `rag/nutrition_vqa_rag.py` | Main RAG pipeline (retrieve + GPT-4V) |
| `rag/prompts.py` | System and user prompts |
| `run_rag.py` | CLI and demo script |


# To run RAG locally without billing using LLaVA and OLLAMA
 https://ollama.com install OLLAMA from this website 
