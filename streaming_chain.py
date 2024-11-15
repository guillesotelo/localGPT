from queue import Queue
from threading import Thread
from langchain.chains import LLMChain
from streaming_handler import StreamingHandler  # Import the StreamingHandler class

class StreamingChain:
    def __init__(self, llm, prompt):
        # Create Langchain prompt and LLM chain
        self.llm_chain = LLMChain(llm=llm, prompt=prompt)
        self.thread = None

    def stream(self, input_data):
        queue = Queue()
        handler = StreamingHandler(queue)

        # Run the LLMChain with the handler for streaming
        def task():
            # Execute the chain with callbacks
            self.llm_chain(input_data, callbacks=[handler])

        self.thread = Thread(target=task)
        self.thread.start()

        try:
            while True:
                token = queue.get()
                if token is None:
                    break  # End of stream
                yield token
        finally:
            self.cleanup()

    def cleanup(self):
        if self.thread and self.thread.is_alive():
            self.thread.join()
