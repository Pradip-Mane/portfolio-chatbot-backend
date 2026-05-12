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

@app.post("/chat")
def chat(req: ChatRequest):

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": "You are Pradip's portfolio AI assistant."},
            {"role": "user", "content": req.message}
        ]
    )

    return {"reply": response.output_text}