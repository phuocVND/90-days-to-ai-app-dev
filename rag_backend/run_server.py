import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


# import os
# import re
# import fitz  # PyMuPDF

# class PDFReader:
#     def __init__(self, folder):
#         self.folder = folder

#     def _clean_text(self, text):
#         # Chuẩn hóa xuống dòng và khoảng trắng
#         text = text.replace('\xa0', ' ')  # loại bỏ ký tự space đặc biệt
#         text = text.replace('\r', ' ').replace('\n', ' ')
#         text = re.sub(r'\s+', ' ', text)
#         # Xóa ký tự không ASCII
#         text = re.sub(r'[^\x00-\x7F]+', ' ', text)
#         return text.strip()

#     def _read_all_pdfs(self):
#         text = ""
#         for filename in os.listdir(self.folder):
#             if filename.lower().endswith(".pdf"):
#                 pdf_path = os.path.join(self.folder, filename)
#                 print(f"📖 Reading file: {pdf_path}")
#                 with fitz.open(pdf_path) as doc:
#                     for i, page in enumerate(doc):
#                         page_text = page.get_text("text")
#                         if page_text.strip():
#                             text += page_text.strip() + " "
#                         else:
#                             print(f"⚠️ Warning: Page {i+1} of {filename} has no text.")
#         return text

#     def _split_into_chunks(self, text, max_sentences=1, max_words=2024):
#         text = self._clean_text(text)

#         # Regex mạnh hơn: chia theo . ! ? nhưng vẫn giữ dấu chấm trong câu viết tắt
#         sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
#         chunks, current_chunk, word_count = [], [], 0

#         for sentence in sentences:
#             words = sentence.split()
#             if len(current_chunk) >= max_sentences or (word_count + len(words)) > max_words:
#                 chunks.append(" ".join(current_chunk))
#                 current_chunk, word_count = [], 0

#             current_chunk.append(sentence.strip())
#             word_count += len(words)

#         if current_chunk:
#             chunks.append(" ".join(current_chunk))

#         return chunks


# if __name__ == "__main__":
#     folder_path = "data"
#     reader = PDFReader(folder_path)
#     all_text = reader._read_all_pdfs()

#     chunks = reader._split_into_chunks(all_text, max_sentences=1, max_words=512)

#     print(f"📦 Total chunks created: {len(chunks)}")
#     print("\n🧩 Example chunks:\n")
#     for i, chunk in enumerate(chunks[:5]):  # In 5 chunk đầu
#         print(f"--- Chunk {i+1} ---")
#         print(chunk[:400], "\n")
