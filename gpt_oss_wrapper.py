from langchain.llms.base import LLM
from langchain.schema import LLMResult
from pydantic import BaseModel
from transformers import TextIteratorStreamer
import threading
from typing import Any

class GPTOSSWrapper(LLM):
    model: Any
    tokenizer: Any
    generation_config: Any

    def __init__(self, model, tokenizer, generation_config, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'model', model)
        object.__setattr__(self, 'tokenizer', tokenizer)
        object.__setattr__(self, 'generation_config', generation_config)

    @property
    def _llm_type(self):
        return "gpt-oss"

    def _call(self, prompt: str, stop=None) -> str:
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.generation_config.max_new_tokens,
            temperature=self.generation_config.temperature,
            top_p=self.generation_config.top_p,
            repetition_penalty=self.generation_config.repetition_penalty
        )
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        return "".join([t for t in streamer])

    async def _acall(self, prompt: str, stop=None) -> str:
        return self._call(prompt, stop=stop)
