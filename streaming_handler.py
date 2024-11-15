import logging
from queue import Queue
from langchain.callbacks.base import BaseCallbackHandler

class StreamingHandler(BaseCallbackHandler):
    def __init__(self, queue):
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs):
        """This method will be called every time a new token is generated."""
        self.queue.put(token)

    def on_llm_end(self, response, **kwargs):
        """This method will be called when the response is finished."""
        self.queue.put(None)  # Signal that the stream has ended

    def on_llm_error(self, error, **kwargs):
        """This method will be called in case of an error."""
        logging.error(f"Error in LLM: {error}")
        self.queue.put(None)  # Signal that the stream has ended in case of an error
