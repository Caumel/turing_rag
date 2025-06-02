from typing import Dict
from gradio import ChatInterface, Chatbot, themes, MultimodalTextbox
from langchain_community.document_loaders import PyMuPDFLoader
import os
import shutil

from dotenv import load_dotenv
load_dotenv()

from rag_agent import RAGAgent

rag_agent = RAGAgent()

def process_message(message: Dict, history):
    try:
        if message["text"] and message['text'] != "":
            response = ""
            last_step = 0
            for msg in rag_agent.process_message(message['text']):
                if last_step < msg["last_step"]:
                    last_step = msg["last_step"]
                    response = ""
                    yield response
                response += str(msg["response"])
                yield response
        if message['files'] and len(message['files']) > 0:
                try:
                    for file in message['files']:
                        filename = os.path.basename(file)
                        new_path = os.path.join("docs", filename)
                        shutil.move(file, new_path)
                        docs = PyMuPDFLoader(new_path).load()
                    yield "Archivos almacenados, ¿en qué puedo ayudarte?"
                except Exception as e:
                    print(e)
                    yield "Error almacenando archivos"
    except Exception as e:
        print(e)
    
history = [
    {"role": "assistant", "content": "Hola, soy un asistente que te ayudará en tus consultas sobre los PDFs, ¿Cómo te puedo ayudar?"},]
demo = ChatInterface(
    chatbot=Chatbot(history, type="messages", height="80%", resizable=True),
    theme=themes.Default(),
    title="PDF",
    editable=True,
    save_history=True,
    fn=process_message,
    type="messages",
    multimodal=True,
    fill_height=True,
    fill_width=True,
    textbox=MultimodalTextbox(
        interactive=True,
        autoscroll=True,
        container=False,
        stop_btn=True,
        autofocus=True,
    ),
)

# Launch the chatbot
demo.launch(
    share=False,
    server_port=7851
    )