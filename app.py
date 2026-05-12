from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import pdfplumber
from bs4 import BeautifulSoup

def load_resume():
    text = ""
    with pdfplumber.open("resume.pdf") as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

def load_html():
    with open("portfolio.html", "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")


def get_context():
    resume_text = load_resume()
    website_text = load_html()

    return f"""
RESUME:
{resume_text}

WEBSITE:
{website_text}
"""


app = FastAPI()

# CORS (allow your website)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"status": "Chatbot backend is running 🚀"}

@app.post("/chat")
def chat(req: ChatRequest):

    context = get_context()

    system_prompt = f"""
You are Pradip Mane's AI portfolio assistant.

Use ONLY the information below to answer questions.
If answer is not in data, say "Not available in portfolio".

--- DATA ---
{context}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.message}
        ]
    )

    return {"reply": response.output_text}
