from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os

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
    system_prompt = """
    You are Pradip Mane's portfolio AI assistant.
    Answer questions about Pradip's portfolio, skills, projects, resume, and experience.
    Be professional, short, and helpful.
    If you do not know something, say it clearly.
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.message}
        ]
    )

    return {"reply": response.output_text}
