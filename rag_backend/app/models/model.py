import os
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import PyPDF2
import re

class PDFModel:
    def __init__(self, folder="data"):
        self.folder = folder
        self.embed_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        self.qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
        self.chunks, self.chunk_embeddings = self._prepare_data()

    def _read_all_pdfs(self):
        text = ""
        for filename in os.listdir(self.folder):
            if filename.lower().endswith(".pdf"):
                with open(os.path.join(self.folder, filename), "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
        return text

    def _split_into_chunks(self, text, max_sentences=5, max_words=1024):
        sentences = re.split(r'(?<=[. ! ? ])\s+', text.strip())
        chunks, current_chunk, word_count = [], [], 0

        for sentence in sentences:
            words = sentence.split()
            if len(current_chunk) >= max_sentences or (word_count + len(words)) > max_words:
                chunks.append(" ".join(current_chunk))
                current_chunk, word_count = [], 0

            current_chunk.append(sentence)
            word_count += len(words)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


    def _prepare_data(self):
        text = self._read_all_pdfs()
        chunks = self._split_into_chunks(text)
        embeddings = self.embed_model.encode(chunks, show_progress_bar=True)
        return chunks, embeddings

    def get_top_chunks(self, question, top_k=2):
        q_emb = self.embed_model.encode([question])[0]
        sims = [np.dot(q_emb, c_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(c_emb)) for c_emb in self.chunk_embeddings]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [self.chunks[i] for i in top_idx]
    
    def build_prompt(self, question, top_k):
        context = "\n".join(self.get_top_chunks(question, top_k))
        prompt = f"""
        You are a knowledgeable assistant. Please answer the following question based strictly on the reference document provided below. 
        If the answer is not found, say "The information is not available in the provided document.

        Question:
        {question}

        Below is the reference document. 
        Use the information in it to answer the question above accurately and concisely.

        Reference Document:
        ----------------------------------------
        {context}
        ----------------------------------------
        """

        return prompt
    
    def answer_question(self, prompt):
        return self.qa_pipeline(prompt, max_length=256)[0]['generated_text']
