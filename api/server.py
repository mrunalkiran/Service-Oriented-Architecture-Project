from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import uvicorn

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

# -----------------------------
# 1. Load environment variables
# -----------------------------
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

print("OPENAI_API_KEY:", bool(os.getenv("OPENAI_API_KEY")))
print("ANTHROPIC_API_KEY:", bool(os.getenv("ANTHROPIC_API_KEY")))
print("GROQ_API_KEY:", bool(os.getenv("GROQ_API_KEY")))

# -----------------------------
# 2. FastAPI app
# -----------------------------
app = FastAPI(
    title="Meta Search LLM Server",
    version="1.0",
    description="Ask any question and get answers from OpenAI, Claude, Groq, and Ollama",
)

# Simple root route so we can see if THIS server is running
@app.get("/")
def root():
    return {"message": "Hello from meta-qa server"}

# -----------------------------
# 3. Request body model
# -----------------------------
class Question(BaseModel):
    question: str

# -----------------------------
# 4. Prompt + parser
# -----------------------------
meta_prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant.
Answer the user's question clearly and concisely in a few paragraphs.

Question: {question}
"""
)
parser = StrOutputParser()

# -----------------------------
# 5. Define models
# -----------------------------
openai_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
)

claude_model = ChatAnthropic(
    model="claude-3-haiku-20240307",
    temperature=0.3,
    max_tokens=512,
)

groq_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=512,
)

# You have llama3.2 in Ollama
ollama_model = OllamaLLM(
    model="llama3.2",
    temperature=0.3,
)

# -----------------------------
# 6. Helper to run each model safely
# -----------------------------
def run_chain(model_name: str, question: str) -> str:
    try:
        model_map = {
            "openai": openai_model,
            "claude": claude_model,
            "groq": groq_model,
            "ollama": ollama_model,
        }
        chain = meta_prompt | model_map[model_name] | parser
        return chain.invoke({"question": question})
    except Exception as e:
        return f"[{model_name} error: {e}]"

# -----------------------------
# 7. /meta-qa endpoint
# -----------------------------
@app.post("/meta-qa")
def meta_qa(body: Question):
    q = body.question

    return {
        "openai": run_chain("openai", q),
        "claude": run_chain("claude", q),
        "groq": run_chain("groq", q),
        "ollama": run_chain("ollama", q),
    }

# -----------------------------
# 8. Run server
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

