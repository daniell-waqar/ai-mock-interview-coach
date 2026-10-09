# AI Mock Interview Coach
A voice-enabled mock interview chatbot built with LangChain. Pick a role and level; the bot asks 5 questions, scores each answer, and ends with a final report with a study plan. Speak your answers with the mic and hear the interviewer's replies.

**Live demo:** <your-render-url>

## Features
- Role and level based interviewer persona (prompt engineering)
- Per-session conversation memory (RunnableWithMessageHistory)
- Streaming responses, voice input and text-to-speech (Web Speech API)
- Runs fully local with Ollama, or deployed with a hosted Llama model

## Run locally (Ollama)
1. `ollama pull llama3.1`
2. `pip install -r requirements.txt`
3. `uvicorn app:app --reload` then open http://localhost:8000

## Deploy (Render + Groq)
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Environment variable: `GROQ_API_KEY` (free key from console.groq.com)

## Stack
Python, FastAPI, LangChain, Ollama, Groq (Llama 3.1), HTML/CSS/JavaScript
