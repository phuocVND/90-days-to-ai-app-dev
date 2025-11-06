from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3

class AnswerResponse(BaseModel):
    question: str
    prompt: str
    answer: str
