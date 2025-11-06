from fastapi import APIRouter
from app.schemas import QuestionRequest, AnswerResponse
from app.services import generate_answer, get_prompt

router = APIRouter(prefix="/api")

@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    prompt = get_prompt(request.question, request.top_k)
    answer = generate_answer(prompt)
    return AnswerResponse(question=request.question, prompt=prompt, answer=answer)
