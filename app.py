from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
import numpy as np
import faiss
import pdfplumber
from bs4 import BeautifulSoup
from openai import OpenAI

# -----------------------------
# INIT APP
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# OPENAI CLIENT
# -----------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# CONFIG
# -----------------------------
DIMENSION = 1536

# ⚠️ SAFE MODE (NO FILE READ FOR FAISS INDEX)
index = faiss.IndexFlatL2(DIMENSION)
metadata = []

# -----------------------------
# REQUEST MODEL
# -----------------------------
class ChatRequest(BaseModel):
    message: str

# -----------------------------
# LOAD RESUME
# -----------------------------
def load_resume():
    text = ""
    with pdfplumber.open("resume.pdf") as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# -----------------------------
# LOAD WEBSITE HTML
# -----------------------------
def load_html():
    with open("index.html", "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    return soup.get_text()

# -----------------------------
# TEXT CHUNKING
# -----------------------------
def chunk_text(text, size=400):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# -----------------------------
# EMBEDDINGS
# -----------------------------
def embed(text):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return res.data[0].embedding

# -----------------------------
# BUILD VECTOR DB (SAFE)
# -----------------------------
def build_index():
    global index, metadata

    print("🚀 Building FAISS index...")

    text = load_resume() + "\n" + load_html()
    chunks = chunk_text(text)

    vectors = []

    for chunk in chunks:
        vec = embed(chunk)
        vectors.append(vec)
        metadata.append(chunk)

    vectors = np.array(vectors).astype("float32")
    index.add(vectors)

    print("✅ FAISS index ready")

# -----------------------------
# SEARCH FUNCTION
# -----------------------------
def search(query):
    if len(metadata) == 0:
        return "No data available"

    q_vec = np.array(embed(query)).astype("float32").reshape(1, -1)

    _, I = index.search(q_vec, k=3)

    results = []
    for i in I[0]:
        if i < len(metadata):
            results.append(metadata[i])

    return "\n".join(results)

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/")
def home():
    return {"status": "Agentic Portfolio AI is running 🚀"}

# -----------------------------
# CHAT ENDPOINT
# -----------------------------
@app.post("/chat")
def chat(req: ChatRequest):

    context = search(req.message)

    system_prompt = f"""
You are Pradip Mane's AI portfolio assistant.

Use ONLY the context below:

{context}

Rules:
- Be short and professional
- If not found, say "Not available in portfolio data"
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.message}
        ]
    )

    return {"reply": response.output_text}

# -----------------------------
# INIT ON STARTUP
# -----------------------------
build_index()
