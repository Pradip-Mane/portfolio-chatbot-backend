from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
import json
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
# PATHS
# -----------------------------
INDEX_PATH = "faiss.index"
META_PATH = "metadata.json"

DIMENSION = 1536

# -----------------------------
# LOAD OR CREATE FAISS INDEX
# -----------------------------
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
else:
    index = faiss.IndexFlatL2(DIMENSION)

if os.path.exists(META_PATH):
    with open(META_PATH, "r") as f:
        metadata = json.load(f)
else:
    metadata = []

# -----------------------------
# REQUEST MODEL
# -----------------------------
class ChatRequest(BaseModel):
    message: str

# -----------------------------
# LOAD RESUME (PDF)
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
    with open("portfolio.html", "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    return soup.get_text()

# -----------------------------
# CHUNK TEXT
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
# BUILD FAISS INDEX (ONCE)
# -----------------------------
def build_index_if_empty():
    global index, metadata

    if len(metadata) > 0:
        return

    print("Building FAISS index...")

    text = load_resume() + "\n" + load_html()
    chunks = chunk_text(text)

    vectors = []

    for c in chunks:
        vectors.append(embed(c))
        metadata.append(c)

    vectors = np.array(vectors).astype("float32")

    index.add(vectors)

    os.makedirs("storage", exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w") as f:
        json.dump(metadata, f)

    print("FAISS index built & saved.")

# -----------------------------
# SEARCH FUNCTION
# -----------------------------
def search(query):
    q_vec = np.array(embed(query)).astype("float32").reshape(1, -1)

    _, I = index.search(q_vec, k=3)

    return "\n".join([metadata[i] for i in I[0]])

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/")
def home():
    return {"status": "Agentic Portfolio AI is running 🚀"}

# -----------------------------
# CHAT ENDPOINT (MAIN BRAIN)
# -----------------------------
@app.post("/chat")
def chat(req: ChatRequest):

    context = search(req.message)

    system_prompt = f"""
You are Pradip Mane's AI portfolio assistant.

Use ONLY the context below to answer questions:

{context}

If answer is not found, say "Not available in portfolio data".
Be professional, short, and accurate.
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
# INIT ON START
# -----------------------------
build_index_if_empty()
