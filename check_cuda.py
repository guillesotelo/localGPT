# from langchain_community.llms import LlamaCpp
import torch
from langchain_community.llms import LlamaCpp
import logging
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response

from load_models import (
    load_quantized_model_awq,
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)

from constants import (
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    N_BATCH,
    CONTEXT_WINDOW_SIZE,
    TEMPERATURE,
    R_PENALTY,
    N_GPU_LAYERS,
    TOP_P,
    TOP_K
)
print("Is CUDA available? ", torch.cuda.is_available())
print("Current CUDA device: ", torch.cuda.get_device_name(torch.cuda.current_device()))

if MODEL_BASENAME:
    if ".gguf" in MODEL_BASENAME.lower() or ".ggml" in MODEL_BASENAME.lower():
        model_path, tokenizer = load_quantized_model_gguf_ggml(MODEL_ID, MODEL_BASENAME, device_type='cuda', logging=logging)
    elif ".awq" in MODEL_BASENAME.lower():
        model_path, tokenizer = load_quantized_model_awq(MODEL_ID, logging)
    else:
        model_path, tokenizer = load_quantized_model_qptq(MODEL_ID, MODEL_BASENAME, device_type='cuda', logging=logging)
else:
    model_path, tokenizer = load_full_model(MODEL_ID, MODEL_BASENAME, device_type='cuda', logging=logging)

model_kwargs = {
        'precision': "fp16",
        'tfs_z': 1.0,
        'offload_kqv': True
    }

model = LlamaCpp(
    model_path=model_path,
    max_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    n_ctx=CONTEXT_WINDOW_SIZE,
    n_batch=N_BATCH,
    callbacks=[StreamingStdOutCallbackHandler()],
    streaming=True,
    repeat_penalty=R_PENALTY,
    n_gpu_layers=N_GPU_LAYERS,
    top_p=TOP_P,
    top_k=TOP_K,
    verbose=True,
    device='cuda',
    model_kwargs=model_kwargs
)

prompt = "Write a short poem about a cat."

output = model.generate([prompt])
print('\n')
print("Raw output:", output)
del model