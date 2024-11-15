from huggingface_hub import cached_download
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

print("hf_hub_download function is available!")
# Test loading a simple model
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully.")
