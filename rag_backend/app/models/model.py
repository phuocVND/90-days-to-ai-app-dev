import os
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import fitz  # PyMuPDF
import os
import re
import textwrap

class PDFModel:
    def __init__(self, folder="data"):
        self.max_sentences = 1
        self.max_words = 1024
        self.folder = folder
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.qa_pipeline = pipeline(
            "text2text-generation",
            model="bigscience/bloomz-560m",
            device=-1  # -1 = CPU
        ) 
        self.chunks, self.chunk_embeddings = self._prepare_data()

        
    def _clean_text(self, text):
        # Chuẩn hóa xuống dòng và khoảng trắng
        text = text.replace('\xa0', ' ')  # loại bỏ ký tự space đặc biệt
        text = text.replace('\r', ' ').replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        # Xóa ký tự không ASCII
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text.strip()

    def _read_all_pdfs(self):
        text = ""
        for filename in os.listdir(self.folder):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(self.folder, filename)
                print(f"📖 Reading file: {pdf_path}")
                with fitz.open(pdf_path) as doc:
                    for i, page in enumerate(doc):
                        page_text = page.get_text("text")
                        if page_text.strip():
                            text += page_text.strip() + " "
                        else:
                            print(f"⚠️ Warning: Page {i+1} of {filename} has no text.")
        return text

    def _split_into_chunks(self, text, max_sentences, max_words):
        text = self._clean_text(text)

        # Regex mạnh hơn: chia theo . ! ? nhưng vẫn giữ dấu chấm trong câu viết tắt
        sentences = re.split(r'(?<=[.!])\s+(?=[A-Z0-9])', text)
        chunks, current_chunk, word_count = [], [], 0

        for sentence in sentences:
            words = sentence.split()
            if len(current_chunk) >= max_sentences or (word_count + len(words)) > max_words:
                chunks.append(" ".join(current_chunk))
                current_chunk, word_count = [], 0

            current_chunk.append(sentence.strip())
            word_count += len(words)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


    def _prepare_data(self):
        text = self._read_all_pdfs()
        chunks = self._split_into_chunks(text, self.max_sentences, self.max_words)
        embeddings = self.embed_model.encode(chunks, show_progress_bar=True)
        return chunks, embeddings

    def get_top_chunks(self, question, top_k=2):
        q_emb = self.embed_model.encode([question])[0]
        sims = [np.dot(q_emb, c_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(c_emb)) for c_emb in self.chunk_embeddings]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [self.chunks[i] for i in top_idx]
    
    def build_prompt(self, question, top_k):
        context = "\n".join(self.get_top_chunks(question, top_k))
        prompt = (
            f"Question:\n{question}\n\n"
            "Please answer the following question based strictly on the reference document provided below.\n\n"
            f"Reference document:\n{context}\n"
        )
        return prompt
    
    def answer_question(self, prompt):
        return self.qa_pipeline(
            prompt,
            max_new_tokens=512,  # sinh tối đa 256 token mới
            temperature=0.3,     # giảm tính sáng tạo, tăng độ chính xác
            top_p=0.9,           # lọc token ít khả năng
            do_sample=True        # sampling để có câu trả lời linh hoạt hơn
        )[0]['generated_text']

