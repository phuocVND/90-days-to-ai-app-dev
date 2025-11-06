from app.models.model import PDFModel

pdf_model = PDFModel()


def get_prompt(question: str, top_k: int = 2) -> str:
    return pdf_model.build_prompt(question, top_k)
def generate_answer(prompt) -> str:
    # print(prompt, flush=True)
    return pdf_model.answer_question(prompt)