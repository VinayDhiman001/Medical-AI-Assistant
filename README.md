# Medical AI Assistant

An end-to-end Medical AI Assistant powered by fine-tuned Mistral 7B with RAG pipeline.

## Features
- Fine-tuned Mistral 7B on MedQA dataset
- RAG pipeline for document-based answers
- FastAPI REST API
- Streamlit UI

## Tech Stack
- Model: Mistral 7B (QLoRA fine-tuned)
- RAG: LangChain + FAISS + Sentence Transformers
- API: FastAPI
- UI: Streamlit
- Training: HuggingFace PEFT + TRL

## Project Structure
- src/ - All Python classes
- api/ - FastAPI code
- frontend/ - Streamlit UI
- configs/ - Config files

## How to Run
1. Install dependencies: pip install -r requirements.txt
2. Run API: uvicorn api.main:app --reload
3. Run UI: streamlit run frontend/app.py

## Author
Vinay Dhiman
