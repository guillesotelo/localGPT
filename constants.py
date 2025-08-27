import os
from chromadb.config import Settings
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader, UnstructuredFileLoader, UnstructuredMarkdownLoader, UnstructuredHTMLLoader
import json
from datetime import datetime, timezone

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
# SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"
SOURCE_DIRECTORY = os.getenv("SOURCE_DIRECTORY", "/var/lib/hpchatbot/latest")
AUX_DOCS = '/chatbot/AUX_DOCS'
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"
MODELS_PATH = "./models"

if not os.path.exists(PERSIST_DIRECTORY):
    DB_DATE = None
else:
    DB_DATE = datetime.fromtimestamp(os.stat(PERSIST_DIRECTORY).st_mtime, tz=timezone.utc)

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

# MODEL LOADING
INGEST_THREADS = int(os.getenv("INGEST_THREADS", os.cpu_count() or 8))
CONTEXT_WINDOW_SIZE = int(os.getenv("CONTEXT_WINDOW_SIZE", 4096)) # 4096 working, 5120 too
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 1024)) # 2048 with window 4096
TEMPERATURE=float(os.getenv("TEMPERATURE", 0.1))
R_PENALTY=float(os.getenv("R_PENALTY", 1.1))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", -1)) # This should be 0 for CPU use
N_BATCH = int(os.getenv("N_BATCH", 16))
TOP_P = float(os.getenv("TOP_P", 0.9))
TOP_K = int(os.getenv("TOP_K", 20))

# EMBEDDINGS
# SPLIT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
SPLIT_SEPARATORS = ["\n\n", "\n", ". "]
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1280)) # 1280
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 320)) # 320
FETCH_K_DOCS = int(os.getenv("FETCH_K_DOCS", 50))
LAMBDA_MULT = float(os.getenv("LAMBDA_MULT", 0.25))
RETRIEVE_K_DOCS = int(os.getenv("RETRIEVE_K_DOCS", 7))
COLLECTION_METADATA = {"hnsw:space": "cosine"}


# https://python.langchain.com/en/latest/_modules/langchain/document_loaders/excel.html#UnstructuredExcelLoader
DOCUMENT_MAP = {
    ".html": UnstructuredHTMLLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".py": TextLoader,
    # ".pdf": PDFMinerLoader,
    ".pdf": UnstructuredFileLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}

# MODEL_NAME='gpt-oss'
# MODEL_ID = "openai/gpt-oss-20b"
# MODEL_BASENAME = None   # <- not needed for Transformers

# MODEL_NAME='mistral'
# MODEL_ID = os.getenv("MODEL_ID", "bartowski/Mistral-7B-Instruct-v0.3-GGUF")
# MODEL_BASENAME =  os.getenv("MODEL_BASENAME", "Mistral-7B-Instruct-v0.3-Q5_K_M.gguf")

MODEL_NAME='qwen'
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct-GGUF")
MODEL_BASENAME = "/chatbot/source/api/models/models--Qwen--Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q5_k_m.gguf"
# Using this script to download both files that will be merge on model execution by HuggingFace CLI:
# huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
#     --include "qwen2.5-7b-instruct-q5_k_m*.gguf" \
#     --local-dir models \
#     --local-dir-use-symlinks False


EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3") # size 8192
# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/e5-large-v2") # size 1024

# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5") # Uses ~5 GB of VRAM (High Accuracy & Retrieval)
# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "Alibaba-NLP/gte-large-en-v1.5")
# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "allenai/longformer-base-4096")
# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "jinaai/jina-embeddings-v2-base-en")
# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "google/bigbird-pegasus-large-arxiv")

# Default Instructor Model
# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5") # From PrivateGPT
# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "hkunlp/instructor-large") # (Working) Uses 1.5 GB of VRAM (High Accuracy with lower VRAM usage)
# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "hkunlp/instructor-xl") # Uses 5 GB of VRAM (Most Accurate of all models)

# EMBEDDING_MODEL_NAME = "hkunlp/instructor-xl" # Uses 5 GB of VRAM (Most Accurate of all models)
# EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2" # Uses 1.5 GB of VRAM (A little less accurate than instructor-large)
# EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2" # Uses 0.5 GB of VRAM (A good model for lower VRAM GPUs)
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Uses 0.2 GB of VRAM (Less accurate but fastest - only requires 150mb of vram) Apparently better for CPU



# MODEL_ID= "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"
# MODEL_BASENAME = "Meta-Llama-3-8B-Instruct.Q6_K.gguf"

####
#### MULTILINGUAL EMBEDDING MODELS
####
# EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large" # Uses 2.5 GB of VRAM
# EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base" # Uses 1.2 GB of VRAM


#### SELECT AN OPEN SOURCE LLM (LARGE LANGUAGE MODEL)
# Select the Model ID and model_basename
# load the LLM for generating Natural Language responses

#### GPU VRAM Memory required for LLM Models (ONLY) by Billion Parameter value (B Model)
#### Does not include VRAM used by Embedding Models - which use an additional 2GB-7GB of VRAM depending on the model.
####
#### (B Model)   (float32)    (float16)    (GPTQ 8bit)         (GPTQ 4bit)
####    7b         28 GB        14 GB       7 GB - 9 GB        3.5 GB - 5 GB
####    13b        52 GB        26 GB       13 GB - 15 GB      6.5 GB - 8 GB
####    32b        130 GB       65 GB       32.5 GB - 35 GB    16.25 GB - 19 GB
####    65b        260.8 GB     130.4 GB    65.2 GB - 67 GB    32.6 GB -  - 35 GB

# MODEL_ID = "TheBloke/Llama-2-7B-Chat-GGML"
# MODEL_BASENAME = "llama-2-7b-chat.ggmlv3.q4_0.bin"

####
#### (FOR GGUF MODELS)
####

# MODEL_ID = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
# MODEL_BASENAME = "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"

# MODEL_ID = "Aryanne/Orca-Mini-3B-gguf"
# MODEL_BASENAME = "q4_0-orca-mini-3b.gguf"

# MODEL_ID = "TheBloke/Llama-2-13b-Chat-GGUF"
# MODEL_BASENAME = "llama-2-13b-chat.Q4_K_M.gguf"

# MODEL_ID = "TheBloke/Llama-2-7b-Chat-GGUF"
# MODEL_BASENAME = "llama-2-7b-chat.Q4_K_M.gguf"

# MODEL_ID = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"
# MODEL_BASENAME = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

# Use mistral to run on hpu
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# LLAMA 3 # use for Apple Silicon
# MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL_BASENAME = None

# LLAMA 3 # use for NVIDIA GPUs
# MODEL_ID = "unsloth/llama-3-8b-bnb-4bit"
# MODEL_BASENAME = None

# MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
# MODEL_BASENAME = "mistral-7b-instruct-v0.1.Q8_0.gguf"

# MODEL_ID = os.getenv("MODEL_ID", "TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
# MODEL_BASENAME =  os.getenv("MODEL_BASENAME", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")

# MODEL_ID = 'bartowski/Meta-Llama-3-70B-Instruct-GGUF'
# MODEL_BASENAME =  os.getenv("MODEL_BASENAME", "Meta-Llama-3-70B-Instruct-Q5_K_M.gguf")

# MODEL_ID = 'TheBloke/Llama-2-13B-GGUF'
# MODEL_BASENAME =  os.getenv("MODEL_BASENAME", "llama-2-13b.Q5_K_M.gguf")

# MODEL_ID = "TheBloke/Llama-2-70b-Chat-GGUF"
# MODEL_BASENAME = "llama-2-70b-chat.Q4_K_M.gguf"

####
#### (FOR HF MODELS)
####

# MODEL_ID = "NousResearch/Llama-2-7b-chat-hf"
# MODEL_BASENAME = None
# MODEL_ID = "TheBloke/vicuna-7B-1.1-HF"
# MODEL_BASENAME = None
# MODEL_ID = "TheBloke/Wizard-Vicuna-7B-Uncensored-HF"
# MODEL_ID = "TheBloke/guanaco-7B-HF"
# MODEL_ID = 'NousResearch/Nous-Hermes-13b' # Requires ~ 23GB VRAM. Using STransformers
# alongside will 100% create OOM on 24GB cards.
# llm = load_model(device_type, model_id=model_id)

####
#### (FOR GPTQ QUANTIZED) Select a llm model based on your GPU and VRAM GB. Does not include Embedding Models VRAM usage.
####

##### 48GB VRAM Graphics Cards (RTX 6000, RTX A6000 and other 48GB VRAM GPUs) #####

### 65b GPTQ LLM Models for 48GB GPUs (*** With best embedding model: hkunlp/instructor-xl ***)
# MODEL_ID = "TheBloke/guanaco-65B-GPTQ"
# MODEL_BASENAME = "model.safetensors"
# MODEL_ID = "TheBloke/Airoboros-65B-GPT4-2.0-GPTQ"
# MODEL_BASENAME = "model.safetensors"
# MODEL_ID = "TheBloke/gpt4-alpaca-lora_mlp-65B-GPTQ"
# MODEL_BASENAME = "model.safetensors"
# MODEL_ID = "TheBloke/Upstage-Llama1-65B-Instruct-GPTQ"
# MODEL_BASENAME = "model.safetensors"

##### 24GB VRAM Graphics Cards (RTX 3090 - RTX 4090 (35% Faster) - RTX A5000 - RTX A5500) #####

### 13b GPTQ Models for 24GB GPUs (*** With best embedding model: hkunlp/instructor-xl ***)
# MODEL_ID = "TheBloke/Wizard-Vicuna-13B-Uncensored-GPTQ"
# MODEL_BASENAME = "Wizard-Vicuna-13B-Uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"
# MODEL_ID = "TheBloke/vicuna-13B-v1.5-GPTQ"
# MODEL_BASENAME = "model.safetensors"
# MODEL_ID = "TheBloke/Nous-Hermes-13B-GPTQ"
# MODEL_BASENAME = "nous-hermes-13b-GPTQ-4bit-128g.no-act.order"
# MODEL_ID = "TheBloke/WizardLM-13B-V1.2-GPTQ"
# MODEL_BASENAME = "gptq_model-4bit-128g.safetensors

### 30b GPTQ Models for 24GB GPUs (*** Requires using intfloat/e5-base-v2 instead of hkunlp/instructor-large as embedding model ***)
# MODEL_ID = "TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ"
# MODEL_BASENAME = "Wizard-Vicuna-30B-Uncensored-GPTQ-4bit--1g.act.order.safetensors"
# MODEL_ID = "TheBloke/WizardLM-30B-Uncensored-GPTQ"
# MODEL_BASENAME = "WizardLM-30B-Uncensored-GPTQ-4bit.act-order.safetensors"

##### 8-10GB VRAM Graphics Cards (RTX 3080 - RTX 3080 Ti - RTX 3070 Ti - 3060 Ti - RTX 2000 Series, Quadro RTX 4000, 5000, 6000) #####
### (*** Requires using intfloat/e5-small-v2 instead of hkunlp/instructor-large as embedding model ***)

### 7b GPTQ Models for 8GB GPUs
# MODEL_ID = "TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ"
# MODEL_BASENAME = "Wizard-Vicuna-7B-Uncensored-GPTQ-4bit-128g.no-act.order.safetensors"
# MODEL_ID = "TheBloke/WizardLM-7B-uncensored-GPTQ"
# MODEL_BASENAME = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"
# MODEL_ID = "TheBloke/wizardLM-7B-GPTQ"
# MODEL_BASENAME = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"

####
#### (FOR GGML) (Quantized cpu+gpu+mps) models - check if they support llama.cpp
####

# MODEL_ID = "TheBloke/wizard-vicuna-13B-GGML"
# MODEL_BASENAME = "wizard-vicuna-13B.ggmlv3.q4_0.bin"
# MODEL_BASENAME = "wizard-vicuna-13B.ggmlv3.q6_K.bin"
# MODEL_BASENAME = "wizard-vicuna-13B.ggmlv3.q2_K.bin"
# MODEL_ID = "TheBloke/orca_mini_3B-GGML"
# MODEL_BASENAME = "orca-mini-3b.ggmlv3.q4_0.bin"

####
#### (FOR AWQ QUANTIZED) Select a llm model based on your GPU and VRAM GB. Does not include Embedding Models VRAM usage.
### (*** MODEL_BASENAME is not actually used but have to contain .awq so the correct model loading is used ***)
### (*** Compute capability 7.5 (sm75) and CUDA Toolkit 11.8+ are required ***)
####
# MODEL_ID = "TheBloke/Llama-2-7B-Chat-AWQ"
# MODEL_BASENAME = "model.safetensors.awq"
