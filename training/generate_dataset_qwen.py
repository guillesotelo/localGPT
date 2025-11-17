import os
import json
from pathlib import Path
from langchain.llms import HuggingFacePipeline  # or your LangChain Qwen wrapper
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import hf_hub_download
from constants import (
    MODEL_ID,
    MODEL_BASENAME,
    MODELS_PATH,
    SOURCE_DIRECTORY,
    AUX_DOCS,
    SERVER_URL
)

dataset_output = "./training/qwen_training_data.jsonl"

# --------------------------
# Set up Qwen LLM in LangChain
# --------------------------

model_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=MODEL_BASENAME,
        resume_download=True,
        cache_dir=MODELS_PATH,
    )

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # uses your GPU
    torch_dtype="auto"
)

text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=1024,
    temperature=0.2,
    top_p=0.95,
)

llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

# --------------------------
# Utility functions
# --------------------------

def read_txt_files(folder_path):
    """Read all txt files and return list of dicts with text + metadata"""
    docs = []
    for file_path in Path(folder_path).glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            docs.append({
                "filename": file_path.name,
                "text": f.read()
            })
    return docs

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks of approx chunk_size words with optional overlap"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# --------------------------
# Generate Q&A using Qwen LLM
# --------------------------

def generate_qa(chunk):
    """
    Generate a factual question + answer pair from a text chunk
    using the Qwen LLM via LangChain.
    """
    prompt = f"""
    You are an expert automotive software engineer. 
    Read the following excerpt and generate 1 concise factual question 
    and its answer strictly based on this text. Do NOT hallucinate.
    
    Excerpt:
    {chunk}
    
    Format:
    Question: ...
    Answer: ...
    """
    
    response = llm(prompt)
    
    # Basic parsing
    text = response.strip()
    question, answer = "",""
    if "Question:" in text and "Answer:" in text:
        try:
            question = text.split("Question:")[1].split("Answer:")[0].strip()
            answer = text.split("Answer:")[1].strip()
        except:
            question, answer = chunk[:50]+"?", chunk[:200]  # fallback
    else:
        # fallback
        question, answer = chunk[:50]+"?", chunk[:200]
    
    return question, answer

# --------------------------
# Convert to Qwen JSONL
# --------------------------

def create_qwen_sample(question, answer, metadata=None):
    """Convert to Qwen JSON chat format"""
    sample = {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
    }
    if metadata:
        sample["metadata"] = metadata
    return sample

def save_jsonl(samples, output_file=dataset_output):
    """Save list of dicts to JSONL file"""
    with open(output_file, "a", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

# --------------------------
# Main pipeline
# --------------------------

def main(input_folder="docs", output_file=dataset_output, chunk_size=500):
    docs = read_txt_files(input_folder)
    all_samples = []
    
    for doc in docs:
        chunks = chunk_text(doc["text"], chunk_size=chunk_size)
        for chunk in chunks:
            question, answer = generate_qa(chunk)
            filename = doc["filename"]
            ext = os.path.splitext(filename)[1]
            if ext not in (".txt", ".md", ".rst"):
                continue
            if '§' in filename:
                spliturl = filename[4:].replace('¤', '/').split('§')
                url_ext = '.md' if ext == '.md' else '.html'
                url = f"[{spliturl[0]}]({SERVER_URL}{spliturl[1].replace(ext, url_ext)})"

            sample = create_qwen_sample(
                question, 
                answer, 
                metadata={"source": url or filename}
            )
            all_samples.append(sample)
    
    save_jsonl(all_samples, output_file)
    print(f"Dataset saved: {output_file}, total samples: {len(all_samples)}")

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    main(input_folder=SOURCE_DIRECTORY, output_file=dataset_output, chunk_size=500)
