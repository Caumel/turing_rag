from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

import tiktoken

class SummarizingChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, llm: ChatOpenAI, token_limit: int = 1000):
        self.messages: list[BaseMessage] = []
        self.llm = llm
        self.token_limit = token_limit
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def add_message(self, message: BaseMessage) -> None:
        print(f"Mensaje agregado al historial: {type(message)} - {message.content}")
        self.messages.append(message)
        self._maybe_summarize()

    def clear(self):
        self.messages = []

    def _maybe_summarize(self):
        token_count = sum(len(self.encoding.encode(m.content)) for m in self.messages if hasattr(m, 'content'))
        print(f"Token count: {token_count}, Token limit: {self.token_limit}")
        if token_count > self.token_limit:
            context = "\n".join([m.content for m in self.messages[-8:] if hasattr(m, 'content')])
            summary = self.llm.invoke(f"Resume esto brevemente:\n{context}")
            self.messages = [SystemMessage(content=f"Resumen: {summary.content}")]

    def get_messages(self) -> list[BaseMessage]:
        return self.messages