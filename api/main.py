from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str
    use_rag: bool = False

class AnswerResponse(BaseModel):
    question: str
    answer: str
    mode: str

app = FastAPI(
    title="Medical AI Assistant API",
    description="AI powered medical question answering system",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "Mistral-7B-Medical"}

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    try:
        if request.use_rag:
            answer = rag.answer(request.question)
            mode = "rag"
        else:
            prompt = f"<s>[INST] {request.question} [/INST]"
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=200)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            mode = "llm"
        return AnswerResponse(question=request.question, answer=answer, mode=mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
