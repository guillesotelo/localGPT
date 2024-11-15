# app.py

import logging
import os
import shutil
import subprocess
import argparse
import sys
import time
from queue import Queue, Empty
from flask_cors import CORS

import torch
from flask import Flask, jsonify, request, Response
from langchain.chains import RetrievalQA
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from run_localGPT import load_model
from prompt_template_utils import get_prompt_template
from langchain.vectorstores import Chroma
from werkzeug.utils import secure_filename
from streaming_chain import StreamingChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from prompt_template_utils import get_chat_prompt
from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME
from threading import Lock
# For enterprise use
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
)

# Determine device type
if torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
elif torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
else: 
    DEVICE_TYPE = "cpu"
logging.info(f"Selected device type: {DEVICE_TYPE}")

SHOW_SOURCES = True
logging.info(f"Running on: {DEVICE_TYPE}")
logging.info(f"Display Source Documents set to: {SHOW_SOURCES}")

# API queue addition
request_lock = Lock()

EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})

# Load the vectorstore
DB = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=EMBEDDINGS,
    client_settings=CHROMA_SETTINGS,
)

RETRIEVER = DB.as_retriever()

# Load the model with streaming support
LLM = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME)


app = Flask(__name__)
CORS(app)

@app.route("/api/prompt_route", methods=["GET", "POST"])
def prompt_route():
    global QA
    global request_lock
    user_prompt = request.form.get("user_prompt")
    use_context = request.form.get("use_context", "true").lower() == "true"

    with request_lock:
        if not user_prompt:
            return "No user prompt received", 400
        
        def generate_stream():
            # Use documents context
            prompt, memory = get_prompt_template(promptTemplate_type="mistral", history=False, user_prompt=user_prompt, use_context=use_context)
            
            if use_context:

                QA = RetrievalQA.from_chain_type(
                    llm=LLM,
                    chain_type="stuff",
                    retriever=RETRIEVER,
                    return_source_documents=SHOW_SOURCES,
                    chain_type_kwargs={
                        "prompt": prompt,
                    },
                )
                print(f"\n\n* Using context from documents *\n\n")
                # question_answer_chain = create_stuff_documents_chain(QA, prompt)
                # rag_chain = create_retrieval_chain(RETRIEVER, question_answer_chain)
        
                try:
                    print("\nAnswer: \n")
                    for token in QA(user_prompt)["result"]:
                        if isinstance(token, dict):
                            token = str(token)
                        print(token, end="")
                        yield token.encode("utf-8")
                    print("\n\n")
                except Exception as e:
                    logging.error(f"Error during streaming: {str(e)}")
                    yield "Error occurred during streaming".encode("utf-8")
                
            # Chat with LLM only
            else:
                print(f"\n\n* Using direct chat with LLM *\n\n")
                input_data = {
                    "context": None,
                    "question": user_prompt 
                }

                try:
                    llm_chain = StreamingChain(LLM, prompt)
                    for token in llm_chain.stream(input_data=input_data):
                        if isinstance(token, dict):
                           token = str(token)
                        yield token.encode("utf-8")
                except Exception as e:
                    logging.error(f"Error during streaming: {str(e)}")
                    yield "Error occurred during streaming".encode("utf-8")


    return Response(generate_stream(), mimetype="text/plain")

# Other routes remain unchanged
@app.route("/api/delete_source", methods=["GET"])
def delete_source_route():
    folder_name = "SOURCE_DOCUMENTS"

    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)

    os.makedirs(folder_name)

    return jsonify({"message": f"Folder '{folder_name}' successfully deleted and recreated."})

@app.route("/api/save_document", methods=["GET", "POST"])
def save_document_route():
    if "document" not in request.files:
        return "No document part", 400
    file = request.files["document"]
    if file.filename == "":
        return "No selected file", 400
    if file:
        filename = secure_filename(file.filename)
        folder_path = "SOURCE_DOCUMENTS"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, filename)
        file.save(file_path)
        return "File saved successfully", 200

@app.route("/api/run_ingest", methods=["GET"])
def run_ingest_route():
    global DB
    global RETRIEVER
    global QA
    try:
        if os.path.exists(PERSIST_DIRECTORY):
            try:
                shutil.rmtree(PERSIST_DIRECTORY)
            except OSError as e:
                print(f"Error: {e.filename} - {e.strerror}.")
        else:
            print("The directory does not exist")

        run_langest_commands = ["python", "ingest.py"]
        if DEVICE_TYPE == "cpu":
            run_langest_commands.append("--device_type")
            run_langest_commands.append(DEVICE_TYPE)

        result = subprocess.run(run_langest_commands, capture_output=True)
        if result.returncode != 0:
            return "Script execution failed: {}".format(result.stderr.decode("utf-8")), 500

        # Reload the vectorstore
        DB = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=EMBEDDINGS,
            client_settings=CHROMA_SETTINGS,
        )
        RETRIEVER = DB.as_retriever()
        prompt, memory = get_prompt_template(promptTemplate_type="llama", history=False)

        QA = RetrievalQA.from_chain_type(
            llm=LLM,
            chain_type="stuff",
            retriever=RETRIEVER,
            return_source_documents=SHOW_SOURCES,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )
        return "Script executed successfully: {}".format(result.stdout.decode("utf-8")), 200
    except Exception as e:
        return f"Error occurred: {str(e)}", 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5110, help="Port to run the API on. Defaults to 5110.")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to run the UI on. Defaults to 127.0.0.1. "
             "Set to 0.0.0.0 to make the UI externally "
             "accessible from other devices.",
    )
    args = parser.parse_args()

    app.run(debug=False, host=args.host, port=args.port, threaded=False)