# from langchain_community.llms import LlamaCpp
import torch
from llama_cpp import Llama

print("Is CUDA available? ", torch.cuda.is_available())
print("Current CUDA device: ", torch.cuda.get_device_name(torch.cuda.current_device()))

model_path = './models/models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/snapshots/3a6fbf4a41a1d52e415a4958cde6856d34b2db93/mistral-7b-instruct-v0.2.Q5_K_M.gguf'

model = Llama(
    model_path,
    n_gpu_layers=-1,
    verbose=True,
    max_tokens=512,
    n_ctx=2048,
    temperature=0.1,
    use_mmap=False
)

query = "Write a song about a fishermen and a lake creature"

print(model.create_chat_completion(
    messages=[{
        "role": "user",
        "content": query
        }]
    ))