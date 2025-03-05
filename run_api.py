# app.py

import logging
import os
import shutil
import subprocess
import argparse
import time
from flask_cors import CORS
from dotenv import load_dotenv
import json
import sqlite3
load_dotenv()

import torch
from flask import Flask, jsonify, request, Response
from langchain.chains import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from run_localGPT import load_model
from prompt_template_utils import get_prompt_template, system_prompt, contextualize_q_system_prompt, get_sources_template
from langchain_community.vectorstores import Chroma
from werkzeug.utils import secure_filename
from streaming_chain import StreamingChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel

from constants import (
    CHROMA_SETTINGS, 
    EMBEDDING_MODEL_NAME, 
    PERSIST_DIRECTORY, 
    MODEL_ID, 
    MODEL_BASENAME, 
    MODEL_NAME, 
    RETRIEVE_K_DOCS,
    COLLECTION_METADATA
)

from threading import Lock
from utils import get_embeddings
# For enterprise use
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.DEBUG)

os.environ["CURL_CA_BUNDLE"] = ""
# os.environ["CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

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
stream_lock = Lock()

# Active requests
active_streams = {}

torch.cuda.empty_cache()
import gc

gc.collect()
# Load embeddings Model
EMBEDDINGS = get_embeddings(DEVICE_TYPE)

# Load the vectorstore
DB = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=EMBEDDINGS,
    client_settings=CHROMA_SETTINGS,
    collection_metadata=COLLECTION_METADATA
)

retriever = DB.as_retriever(
    search_kwargs={"k": RETRIEVE_K_DOCS},
    search_type="similarity",
    similarity_metric="cosine" 
)

# Load the model with streaming support
LLM = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME)
logging.info("LLM loaded.")


app = Flask(__name__)
CORS(app, expose_headers=["Stream-ID"])

API_TOKEN = os.getenv("API_TOKEN")

def require_token(func):
    """Decorator to check API token"""
    def wrapper(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token or token != API_TOKEN:
            return jsonify({"error": "Unauthorized"}), 401
        return func(*args, **kwargs)
    return wrapper


#  ---------- DB Init ----------

DB_FILE = "/chatbot/db/chatbot.db"

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                name TEXT,
                comments TEXT,
                messages TEXT
            )
        """)
        conn.commit()
        
init_db()

#  ---------- API ROUTES ----------


@app.route("/api/prompt_route", methods=["GET", "POST"])
def prompt_route():
    logging.info(">>> Entering /api/prompt_route")
    user_prompt = request.form.get("user_prompt")
    use_context = request.form.get("use_context")
    use_history = request.form.get("use_history")
    stream_id = int(request.form.get("stream_id", str(int(time.time() * 1000))))
    stop = request.form.get("stop", "false").lower() == "true"

    if stop:
        with stream_lock:
            if stream_id not in active_streams:
                return jsonify({"error": f"Stream {stream_id} not found"}), 404
            active_streams[stream_id]["stop"] = True
        logging.info(f"Stop signal sent for stream ID: {stream_id}")
        return jsonify({"message": f"Stream {stream_id} stop signal issued."}), 200

    if use_context is None:
        use_context = True
    else:
        use_context = use_context.lower() == "true"

    if use_history is None:
        use_history = False
    else:
        use_history = use_history.lower() == "true"

    if not user_prompt:
        return "No user prompt received", 400

    with stream_lock:
        if stream_id in active_streams:
            return jsonify({"error": f"Stream {stream_id} is already in progress"}), 409
        active_streams[stream_id] = {"stop": False, "lock": Lock()}

    def generate_stream():
        try:
            with active_streams[stream_id]["lock"]:
                prompt, memory = get_prompt_template(
                    promptTemplate_type=MODEL_NAME, user_prompt=user_prompt, use_context=use_context
                )

                # Use documents context
                if use_context:
                    print('\n')
                    print(f"\n\n*** Using chat with CONTEXT ***\n")
                    print('\n')
                    print(f"\nUser Prompt: {user_prompt}")
                    print('\n')

                    ctx_system_prompt = f"""
                        {system_prompt}
                        \n\n
                        {{context}}
                        """
                    if use_history:
                        contextualize_q_prompt = ChatPromptTemplate.from_messages(
                            [
                                ("system", contextualize_q_system_prompt),
                                MessagesPlaceholder("chat_history"),
                                ("human", "{input}"),
                            ]
                        )
                        history_aware_retriever = create_history_aware_retriever(
                            LLM, retriever, contextualize_q_prompt
                        )
                        ctx_history_prompt = ChatPromptTemplate.from_messages(
                            [
                                ("system", ctx_system_prompt),
                                MessagesPlaceholder("chat_history"),
                                ("human", "{input}"),
                            ]
                        )
                        question_answer_chain = create_stuff_documents_chain(LLM, ctx_history_prompt)
                        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
                        chain = rag_chain.pick("answer")
                        
                        for token in chain.stream({"input": user_prompt}):
                            if active_streams.get(stream_id, {}).get("stop"):
                                logging.info(f"Stopping stream {stream_id} due to user request.")
                                break  # Exit the loop if stop signal is received
                            yield token
                        print("\n\n")

                    else:
                        ctx_prompt = ChatPromptTemplate.from_messages(
                            [
                                ("system", ctx_system_prompt),
                                ("human", "{input}"),
                            ]
                        )


                        # Build stream chain
                        question_answer_chain = create_stuff_documents_chain(LLM, prompt)
                        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                        chain = rag_chain.pick("answer")

                        for token in chain.stream({"input": f"query: {user_prompt}"}):
                            if active_streams.get(stream_id, {}).get("stop"):
                                logging.info(f"Stopping stream {stream_id} due to user request.")
                                break  # Exit the loop if stop signal is received
                            yield token
                        print("\n\n")

                        retriever_results_with_scores = DB.similarity_search_with_relevance_scores(user_prompt, k=RETRIEVE_K_DOCS)

                        for doc, score in retriever_results_with_scores:
                            logging.info(f"\n")
                            logging.info(f"Document: {doc.metadata.get('source', 'Unknown Source')} | Score: {score}")

                        SIMILARITY_THRESHOLD = 0.71

                        filtered_results = sorted(
                            [(doc, score) for doc, score in retriever_results_with_scores if score >= SIMILARITY_THRESHOLD],
                            key=lambda x: x[1], 
                            reverse=True
                        )

                        retriever_results = [doc for doc, score in filtered_results]

                        # If sources don't enter the threshold then we check if the slopoe 
                        # between the first two is steep enough to return the first one.
                        if not filtered_results:
                            sorted_results = sorted(retriever_results_with_scores, key=lambda x: x[1], reverse=True)

                            if len(sorted_results) > 1 and (sorted_results[0][1] - sorted_results[1][1]) >= 0.05:
                                retriever_results = [sorted_results[0][0]]

                        sources = []
                        unique_sources = set()
                        for doc in retriever_results:
                            source = doc.metadata.get("source", "Unknown Source")
                            if source not in unique_sources:
                                unique_sources.add(source)
                                sources.append(source)

                        if sources:
                            if len(sources) > 1:
                                yield f"\n <br/><br/><strong>Sources:</strong>"
                            else:
                                yield f"\n <br/><br/><strong>Source:</strong>"
                            for i, source in enumerate(sources[:4]):
                                unreleased_mark = ' (unreleased)' if 'UNRELEASED' in source else ''
                                time.sleep(0.15)  # Simulating streaming effect
                                yield f"\n- {source}{unreleased_mark}"


                # Chat with LLM only
                else:
                    print(f"\n\n*** Using direct chat with LLM ***\n")
                    input_data = {"context": None, "question": user_prompt, "history": use_history}

                    try:
                        print("\nAnswer: \n")
                        llm_chain = StreamingChain(LLM, prompt)
                        for token in llm_chain.stream(input_data=input_data):
                            if active_streams.get(stream_id, {}).get("stop"):
                                logging.info(f"Stopping stream {stream_id} due to user request.")
                                break  # Exit the loop if stop signal is received
                            yield token.encode("utf-8")
                        print("\n\n")
                    except Exception as e:
                        logging.error(f"Error during streaming: {str(e)}")
                        yield "Error occurred during streaming".encode("utf-8")

        finally:
            with stream_lock:
                active_streams.pop(stream_id, None)

    response = Response(generate_stream(), mimetype="text/plain")
    response.headers["Stream-ID"] = stream_id
    logging.info(f">>> Generated response for stream ID: {stream_id}")
    logging.info(">>> Exiting /api/prompt_route")
    return response

# ---- This is a testing route ----
@app.route("/api/prompt_route_test", methods=["GET", "POST"])
def prompt_route_test():
    logging.info(">>> Entering /api/prompt_route_test")
    user_prompt = request.form.get("user_prompt")
    use_context = request.form.get("use_context")
    use_history = request.form.get("use_history")
    stream_id = int(request.form.get("stream_id", str(int(time.time() * 1000))))
    stop = request.form.get("stop", "false").lower() == "true"

    if stop:
        with stream_lock:
            if stream_id not in active_streams:
                return jsonify({"error": f"Stream {stream_id} not found"}), 404
            active_streams[stream_id]["stop"] = True
        logging.info(f"Stop signal sent for stream ID: {stream_id}")
        return jsonify({"message": f"Stream {stream_id} stop signal issued."}), 200

    if use_context is None:
        use_context = True
    else:
        use_context = use_context.lower() == "true"

    if use_history is None:
        use_history = False
    else:
        use_history = use_history.lower() == "true"

    if not user_prompt:
        return "No user prompt received", 400

    with stream_lock:
        if stream_id in active_streams:
            return jsonify({"error": f"Stream {stream_id} is already in progress"}), 409
        active_streams[stream_id] = {"stop": False, "lock": Lock()}

    def generate_stream():
        try:
            with active_streams[stream_id]["lock"]:
                print(f"\n\n*** TESTING CHAT ***\n")
                
                # Step 1: Retrieve relevant documents first (stream retrieval)
                retriever_results = retriever.get_relevant_documents(user_prompt)
                sources = []
                for doc in retriever_results:
                    sources.append(doc.metadata.get("source", "Unknown Source"))
                
                prompt, memory = get_prompt_template(
                    promptTemplate_type=MODEL_NAME, user_prompt=user_prompt, use_context=use_context
                )
                question_answer_chain = create_stuff_documents_chain(LLM, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                chain = rag_chain.pick("answer")

                for token in chain.stream({"input": user_prompt}):
                    if active_streams.get(stream_id, {}).get("stop"):
                        logging.info(f"Stopping stream {stream_id} due to user request.")
                        break  # Exit the loop if stop signal is received
                    yield token
                print("\n\n")

                # Step 4: Stream sources as they are retrieved
                if sources:
                    yield f"\n <br/><br/><strong>Sources:</strong>"
                    for source in sources:
                        yield f"\n- {source}"

        finally:
            with stream_lock:
                active_streams.pop(stream_id, None)

    response = Response(generate_stream(), mimetype="text/plain")
    response.headers["Stream-ID"] = stream_id
    logging.info(f">>> Generated response for stream ID: {stream_id}")
    logging.info(">>> Exiting /api/prompt_route_test")
    return response


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
    global retriever
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
        retriever = DB.as_retriever()

        return "Script executed successfully: {}".format(result.stdout.decode("utf-8")), 200
    except Exception as e:
        return f"Error occurred: {str(e)}", 500


@app.route("/api/health", methods=["GET"])
def check_api_health():
    return "API Status: OK"

# ------ FEEDBACK -------

@app.route("/api/save_feedback", methods=["POST"])
def save_feedback():
    try:
        data = request.get_json()  # Expecting JSON in request body

        if "messages" in data and "id" in data:
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()

                # Get existing columns in the feedback table
                cursor.execute("PRAGMA table_info(feedback)")
                existing_columns = {row[1] for row in cursor.fetchall()}  # row[1] contains column names

                # Expected columns from data
                required_columns = set(data.keys()) - {"messages", "id"}  # Exclude messages (stored as JSON) and id (used by DB)
                required_columns.update(["session_id", "messages", "createdAt"])  # Ensure session_id and messages are always there

                # Add missing columns dynamically
                for column in required_columns:
                    if column not in existing_columns:
                        cursor.execute(f'ALTER TABLE feedback ADD COLUMN {column} TEXT')  # Assuming text type

                # Insert feedback
                cursor.execute(f"""
                    INSERT INTO feedback ({', '.join(required_columns)}) 
                    VALUES ({', '.join(['?'] * len(required_columns))})
                """, [json.dumps(data[col]) if col == "messages" else str(data[col]) for col in required_columns])

                conn.commit()
            return jsonify({"message": "Feedback saved successfully"}), 201

        return jsonify({"error": "Invalid data format"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/get_feedback", methods=["GET"])
@require_token
def get_feedback():
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row  # Allows fetching rows as dictionaries
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM feedback")  # Fetch all columns
            feedback_list = [dict(row) for row in cursor.fetchall()]  # Convert rows to dicts

            # Convert messages field to JSON if it exists
            for feedback in feedback_list:
                if "messages" in feedback and feedback["messages"]:
                    try:
                        feedback["messages"] = json.loads(feedback["messages"])
                    except json.JSONDecodeError:
                        pass  # Keep it as a string if JSON parsing fails

        return jsonify(feedback_list), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/delete_feedback", methods=["DELETE"])
def delete_feedback():
    try:
        data = request.get_json()
        id = data.get("id")

        if not id:
            return jsonify({"error": "id is required"}), 400

        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()

            cursor.execute("DELETE FROM feedback WHERE id = ?", (id,))
            conn.commit()

            if cursor.rowcount == 0:
                return jsonify({"error": "No record found with the given id"}), 404

        return jsonify({"message": "Feedback deleted successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/update_feedback", methods=["PUT"])
def update_feedback():
    try:
        data = request.get_json()
        id = data.get("id")

        if not id:
            return jsonify({"error": "id is required"}), 400

        if len(data.keys()) <= 1:  # Only id is provided, no update fields
            return jsonify({"error": "No fields to update"}), 400

        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()

            # Get existing columns in the feedback table
            cursor.execute("PRAGMA table_info(feedback)")
            existing_columns = {row[1] for row in cursor.fetchall()}

            # Prepare fields to update (excluding id)
            update_columns = {col: data[col] for col in data if col not in {"id"}}

            if not update_columns:
                return jsonify({"error": "No valid columns to update"}), 400

            # Add missing columns dynamically
            for column in update_columns.keys():
                if column not in existing_columns:
                    cursor.execute(f'ALTER TABLE feedback ADD COLUMN {column} TEXT')

            # Generate dynamic SQL query for updating
            set_clause = ", ".join(f"{col} = ?" for col in update_columns.keys())
            values = list(update_columns.values()) + [id]

            cursor.execute(f"""
                UPDATE feedback SET {set_clause} WHERE id = ?
            """, values)

            conn.commit()

            if cursor.rowcount == 0:
                return jsonify({"error": "No record found with the given id"}), 404

        return jsonify({"message": "Feedback updated successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/vectorstore_search', methods=['GET'])
def search_vectors():
    query = request.args.get('query', '')

    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    query_embedding = EMBEDDINGS.embed_query(f"query: {query}")

    # Search ChromaDB for similar vectors
    results =  DB._collection.query(query_embeddings=[query_embedding], n_results=10)
    # Extract matched documents

    matching_docs = results.get("documents", [[]])[0]
    found_exact = any(query.lower() in doc_text.lower() for doc_text in matching_docs)
    
    # Return matches as JSON
    return jsonify({"query": query, "matches": matching_docs, "exact": found_exact})


#  ---------- END OF API ROUTES ----------


# Only serving Flash for development. Produciton runs with gunicorn:
# gunicorn --bind 0.0.0.0:5000 run_api:app --workers 1 --threads 1 --timeout 300

if os.getenv("DEVELOPMENT", False) and __name__ == "__main__":
    print("@@@@@@@@@@@@ RUNNING FLASK APP @@@@@@@@@@@@")
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000, help="Port to run the API on. Defaults to 5110.")
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
