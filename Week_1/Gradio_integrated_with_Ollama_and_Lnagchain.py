# app_langchain_ollama_gradio.py
import gradio as gr
from langchain_community.llms import Ollama   # or use the official langchain_ollama import if available
from langchain_core.messages import HumanMessage, AIMessage

# instantiate the LangChain wrapper for Ollama
llm = Ollama(model="llama2")  # check exact class path for your installed package/version

def build_prompt(history, message):
    """Combine history + current message into a single prompt string."""
    conversation = ""
    for h in history:
        role = "User" if h["role"] == "user" else "Assistant"
        conversation += f"{role}: {h['content']}\n"
    conversation += f"User: {message}\nAssistant:"
    return conversation

def predict(message, history):
    try:
        prompt = build_prompt(history, message)
        response = llm.invoke(prompt)   # for LangChain 1.x, .invoke() expects a string
        return str(response)
    except Exception as e:
        # show the actual error text so you can debug
        return f"⚠️ Error: {e}"

demo = gr.ChatInterface(
    fn=predict,
    type="messages",
    title="LangChain + Ollama + Gradio",
    description="Chat with a local Ollama model through LangChain."
)
demo.launch(share=True)
