import os
from typing import List
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableWithMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import Tool, initialize_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.schema import Document
from langchain_core.runnables import RunnableLambda

from summery import SummarizingChatMessageHistory

class RAGAgent:
    def __init__(self, pdf_folder: str = "docs"):
        self.llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-3.5-turbo",
            temperature=0,
            streaming=True
        )
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = self.create_vectorstore_from_pdfs(pdf_folder)
        self.histories = {}

        ########################
        # Memory and Summery conversation
        base_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_type="similarity"),
            return_source_documents=True
        )

        chain = (
            RunnableLambda(lambda x: base_chain.invoke(x))
            | RunnableLambda(lambda result: {
                "output": result["answer"],
            })
        )

        def get_history(session_id: str):
            if session_id not in self.histories:
                self.histories[session_id] = SummarizingChatMessageHistory(llm=self.llm, token_limit=1000)
            return self.histories[session_id]
        
        self.chain = RunnableWithMessageHistory(
            chain,
            get_session_history=get_history,
            input_messages_key="question",
            history_messages_key="chat_history"
        )
        ########################

        self.tools = [
            Tool.from_function(
                func=PythonREPLTool().run,
                name="Python",
                description="Ejecuta código Python para responder preguntas matemáticas o cálculos."
            )
        ]

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent="chat-zero-shot-react-description",
            verbose=True
        )

    def create_vectorstore_from_pdfs(self, folder_path: str) -> FAISS:
        docs: List[Document] = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(folder_path, filename))
                docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = splitter.split_documents(docs)
        return FAISS.from_documents(texts, self.embeddings)

    def detect_code_question(self, query: str) -> bool:
        trigger_keywords = ["calcula", "cuanto es", "\*", "/", "^", "%", "raiz", "potencia", "log", "math"]
        return any(keyword in query.lower() for keyword in trigger_keywords)

    def process_message(self, query: str):
        if self.detect_code_question(query):
            try:
                result = self.agent.invoke(query)
                print(f"Resultado del código: {result}")
                yield {"response": str(result["output"]), "last_step": 1}
            except Exception as e:
                yield {"response": f"Ocurrió un error al ejecutar el código: {str(e)}", "last_step": 1}
        else:
            try:
                full_response = ""
                for chunk in self.chain.stream(
                    {"question": query},
                    config={"configurable": {"session_id": "usuario_1"}}
                ):
                    partial = chunk.get("output", "")
                    full_response += partial
                    yield {
                        "response": full_response,
                        "last_step": 1,
                    }
            except Exception as e:
                yield {"response": f"Ocurrió un error al generar la respuesta: {str(e)}", "last_step": 1}