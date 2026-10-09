"""AI Mock Interview Coach - LangChain (Ollama locally, Groq when deployed). No RAG."""
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

# Provider switch: GROQ_API_KEY set -> hosted Llama (deployment); otherwise local Ollama.
if os.getenv("GROQ_API_KEY"):
    from langchain_groq import ChatGroq
    llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"), temperature=0.7)
else:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2"), temperature=0.7)

SYSTEM_PROMPT = """You are a professional interviewer conducting a mock interview
for a {level} {role} position.

Rules:
1. Ask ONE question at a time. Start with a short greeting, then the first question.
2. After each candidate answer reply in this format:
   Score: X/10
   Strengths: one or two short points
   Improve: one or two short points
   Then ask the next question, slightly harder than the last.
3. After 5 questions, stop and give a Final Report: overall score,
   top 3 strengths, top 3 areas to improve, and a short study plan.
4. Stay in role. If the candidate goes off topic, steer back politely.
5. Keep replies concise and encouraging."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])
chain = prompt | llm | StrOutputParser()

sessions: dict[str, InMemoryChatMessageHistory] = {}

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    return sessions.setdefault(session_id, InMemoryChatMessageHistory())

chat_chain = RunnableWithMessageHistory(
    chain, get_history, input_messages_key="input", history_messages_key="history"
)

class ChatRequest(BaseModel):
    session_id: str
    role: str
    level: str
    message: str

app = FastAPI(title="AI Mock Interview Coach")

@app.post("/chat")
async def chat(req: ChatRequest):
    async def stream():
        try:
            async for token in chat_chain.astream(
                {"role": req.role, "level": req.level, "input": req.message},
                config={"configurable": {"session_id": req.session_id}},
            ):
                yield token
        except Exception:
            yield "\n[Error] The language model is unavailable. Check Ollama (local) or the API key (deployed)."
    return StreamingResponse(stream(), media_type="text/plain")

@app.post("/reset/{session_id}")
async def reset(session_id: str):
    sessions.pop(session_id, None)
    return {"ok": True}

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")