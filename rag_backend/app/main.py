from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from transformers import pipeline

# 1️⃣ Load dữ liệu và chia chunk
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = [p.strip() for p in text.split("\n\n") if p.strip()]

# 2️⃣ Tạo embedding vector cho từng chunk
embed_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
chunk_embeddings = embed_model.encode(chunks)

# 3️⃣ Hàm tính cosine similarity
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 4️⃣ Hàm tìm top chunk liên quan
def get_top_chunks(question, top_k=1):
    q_emb = embed_model.encode([question])[0]
    sims = [cosine_sim(q_emb, c_emb) for c_emb in chunk_embeddings]
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in top_idx]

# 5️⃣ Câu hỏi ví dụ
question = "What role does an inverter play in a solar power system?"

top_chunks = get_top_chunks(question, top_k=1)
context = "\n".join(top_chunks)

# 6️⃣ Tạo prompt
prompt = f"""
Dưới đây là đoạn tài liệu tham khảo:
{context}

Hãy trả lời câu hỏi sau dựa vào đoạn trên:
{question}
"""

# 7️⃣ Dùng Hugging Face pipeline để trả lời
# Model BART nhỏ (free)
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
answer = qa_pipeline(prompt, max_length=200)[0]['generated_text']

print("Prompt:\n", prompt)
print("\nAI trả lời:\n", answer)
