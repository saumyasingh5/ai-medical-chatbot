from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
import numpy as np
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

index = None
chunks = []

@app.get("/api")
def read_root():
    return {"status": "Medical AI Assistant is running"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global index, chunks
    content = await file.read()
    with open("temp.pdf", "wb") as f:
        f.write(content)
    
    text = ""
    with open("temp.pdf", "rb") as f:
        pdf_reader = PdfReader(f)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    
    if not text.strip():
        return {"error": "PDF se text nahi nikal paya"}
    
    words = text.split()
    chunks = [' '.join(words[i:i+300]) for i in range(0, len(words), 300)]
    
    embeddings = embedder.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    return {"message": "PDF processed", "total_chunks": len(chunks)}

@app.post("/ask")
async def ask_question(data: dict):
    global index, chunks
    if index is None:
        return {"answer": "Pehle PDF upload karo"}
    
    question = data.get("question")
    question_emb = embedder.encode([question])
    
    D, I = index.search(np.array(question_emb).astype('float32'), k=2)
    context = " ".join([chunks[i] for i in I[0]])
    
    return {
        "question": question,
        "answer": f"Document ke hisaab se: {context[:500]}..."
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")
