import os
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import PyPDF2

# 1️⃣ Hàm đọc toàn bộ PDF trong thư mục
def read_all_pdfs(folder_path):
    all_text = ""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"📘 Đang đọc: {filename}")
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        all_text += page_text + "\n"
    return all_text

# Đọc tất cả PDF trong thư mục
folder = "data"  # 👉 đổi thành đường dẫn thư mục PDF của bạn
text = read_all_pdfs(folder)

# 2️⃣ Chia văn bản thành các đoạn (chunk)
# chia theo đoạn trống hoặc mỗi 500 từ để giữ ngữ cảnh
def split_into_chunks(text, max_words=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

chunks = split_into_chunks(text)

# 3️⃣ Tạo embedding vector cho từng chunk
print("🔍 Đang tạo embedding...")
embed_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
chunk_embeddings = embed_model.encode(chunks, show_progress_bar=True)

# 4️⃣ Hàm tính cosine similarity
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 5️⃣ Hàm tìm top chunk liên quan
def get_top_chunks(question, top_k=2):
    q_emb = embed_model.encode([question])[0]
    sims = [cosine_sim(q_emb, c_emb) for c_emb in chunk_embeddings]
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in top_idx]

# 6️⃣ Câu hỏi ví dụ
question = "What is the function of an inverter in a solar energy system?"

top_chunks = get_top_chunks(question, top_k=2)
context = "\n".join(top_chunks)

# 7️⃣ Tạo prompt
prompt = f"""
Dưới đây là đoạn tài liệu tham khảo:
{context}

Hãy trả lời câu hỏi sau dựa vào đoạn trên:
{question}
"""

# 8️⃣ Trả lời bằng mô hình Hugging Face
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
answer = qa_pipeline(prompt, max_length=256)[0]['generated_text']

print("\n🧠 Câu hỏi:", question)
print("\n📜 Ngữ cảnh:\n", context[:1000], "...")
print("\n🤖 AI trả lời:\n", answer)
