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
import redis
import random
load_dotenv()

import torch
from flask import Flask, jsonify, request, Response
from langchain.chains.retrieval import create_retrieval_chain
from run_localGPT import load_model
from prompt_template_utils import get_prompt_template
from langchain_community.vectorstores import Chroma
from werkzeug.utils import secure_filename
from streaming_chain import StreamingChain
from langchain.chains.combine_documents import create_stuff_documents_chain


from constants import (
    CHROMA_SETTINGS, 
    EMBEDDING_MODEL_NAME, 
    PERSIST_DIRECTORY, 
    MODEL_ID, 
    MODEL_BASENAME, 
    MODEL_NAME, 
    SEMANTIC_K_DOCS,
    FULLTEXT_K_DOCS,
    COLLECTION_METADATA,
    DB_DATE,
    CONTEXT_WINDOW_SIZE,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    R_PENALTY,
    N_BATCH,
    TOP_P,
    TOP_K,
    SPLIT_SEPARATORS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    FETCH_K_DOCS,
    LAMBDA_MULT,
    SNOK_SYSTEM_PROMPT,
    PERSIST_DIRECTORY_SNOK,
    TECH_ISSUE_LLM,
    CATEGORY_MAP,
    OOS_MESSAGE
)

from hybrid_retriever import (
    HybridRetriever, 
    search_fts,
    quote_fts_token,
    get_uncommon_or_identifier_words
)

from threading import Lock
from utils import (
    get_embeddings,
    document_to_dict,
    StopStreamHandler,
    delete_file_later,
    NonClosingBytesIO
    )
import re
# For enterprise use
import ssl

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
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

# Active requests
r_streams = redis.Redis(host='localhost', port=6379, db=0)
STREAM_TTL = 300

torch.cuda.empty_cache()
import gc

gc.collect()
# Load embeddings Model
EMBEDDINGS = get_embeddings(DEVICE_TYPE)

# Create retrievers
RETRIEVER_MAP = {}
for category in CATEGORY_MAP.keys():
    RETRIEVER_MAP[category] = {}
    category_dir = os.path.join(PERSIST_DIRECTORY, category.lower())
    
    RETRIEVER_MAP[category]['DB'] = Chroma(
        persist_directory=category_dir,
        embedding_function=EMBEDDINGS,
        client_settings=CHROMA_SETTINGS,
        collection_metadata=COLLECTION_METADATA
    )

    RETRIEVER_MAP[category]['semantic_retriever'] = RETRIEVER_MAP[category]['DB'].as_retriever(
        search_type="similarity",
        similarity_metric="cosine",
        search_kwargs={"k": SEMANTIC_K_DOCS}
    )

    RETRIEVER_MAP[category]['hybrid_retriever'] = HybridRetriever(
        semantic_retriever=RETRIEVER_MAP[category]['semantic_retriever'],
        k_bm25=FULLTEXT_K_DOCS,
        k_semantic=SEMANTIC_K_DOCS,
        db_path=f"{category}.db",
        use_scores=True
    )


# LLM load
LLM = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME)
logging.info("LLM ready in API.")


app = Flask(__name__)
CORS(app, expose_headers=["Stream-ID"])

API_TOKEN = os.getenv("API_TOKEN")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")

def require_token(func):
    """Decorator to check API token"""
    def wrapper(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token or (token != API_TOKEN and token != ADMIN_TOKEN):
            return jsonify({"error": "Unauthorized"}), 401
        return func(*args, **kwargs)
    
    # Changing the wrapper name so it doesn't throw errors when called more than once
    wrapper.__name__ = func.__name__
    return wrapper

def require_admin_token(func):
    """Decorator to check API token"""
    def wrapper(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token or token != ADMIN_TOKEN:
            return jsonify({"error": "Unauthorized"}), 401
        return func(*args, **kwargs)
    
    # Changing the wrapper name so it doesn't throw errors when called more than once
    wrapper.__name__ = func.__name__
    return wrapper


#  ---------- DB Init ----------

DB_FILE = "/chatbot/db/chatbot.db"

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()

        # Feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                name TEXT,
                comments TEXT,
                messages TEXT
            )
        """)

        # Analytics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                message_count INTEGER,
                token_count INTEGER,
                duration_seconds INTEGER,
                prompt TEXT,
                messages TEXT
            )
        """)

        # Ensure UNIQUE index exists (safe for existing DBs)
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_analytics_session_id
            ON analytics(session_id)
        """)
        
        conn.commit()
        
init_db()

#  ---------- API ROUTES ----------

@app.route("/api/health", methods=["GET"])
def check_api_health():
    return "API Status: OK"


@app.route("/api/prompt_route", methods=["GET", "POST"])
@require_token
def prompt_route():
    logging.info(">>> Entering /api/prompt_route")
    user_prompt = request.form.get("user_prompt")
    use_context = request.form.get("use_context")
    first_query = request.form.get("first_query")
    use_history = request.form.get("use_history")
    from_source = request.form.get("source")
    stream_id = int(request.form.get("stream_id", str(int(time.time() * 1000))))
    stop = request.form.get("stop", "false").lower() == "true"
    error = ''
    
    if not RETRIEVER_MAP[from_source]:
        print('\n')
        logging.info(f"""
                     
                     Category soruce: {from_source} not found. Falling back to HP
                     
        """)
        from_source = 'HPx'

    if stop:
        logging.info(f"Attempting to stop stream ID: {stream_id}")

        # Check if stream exists
        if not r_streams.exists(f"stream:{stream_id}"):
            return jsonify({"error": f"Stream {stream_id} not found"}), 404

        # Set stop flag (atomic)
        r_streams.hset(f"stream:{stream_id}", "stop", 1)

        logging.info(f"Stop signal sent for stream ID: {stream_id}")
        return jsonify({"message": f"Stream {stream_id} stop signal issued."}), 200

    if use_context is None:
        use_context = True
    else:
        use_context = use_context.lower() == "true"

    if not user_prompt:
        return "No user prompt received", 400
    
    if r_streams.exists(f"stream:{stream_id}"):
        return jsonify({"error": f"Stream {stream_id} is already in progress"}), 409

    r_streams.hset(f"stream:{stream_id}", mapping={"stop": 0})
    r_streams.expire(f"stream:{stream_id}", STREAM_TTL)
    
    def generate_stream():
        nonlocal error
        try:
            r_streams.expire(f"stream:{stream_id}", STREAM_TTL)
            stop_handler = StopStreamHandler(stream_id, r_streams)
            
            # Use documents context
            if use_context:
                logging.info(f"""
                
                Using chat with CONTEXT
                
                User prompt: {user_prompt}
                Source: {from_source}
                First query: {'yes' if first_query else 'no'}
                
                """)

                if from_source == 'SNOK':
                    prompt, memory = get_prompt_template(
                        system_prompt=SNOK_SYSTEM_PROMPT,
                        model_name=MODEL_NAME, 
                        user_prompt=user_prompt, 
                        use_context=use_context
                    )
                else:
                    prompt, memory = get_prompt_template(
                        model_name=MODEL_NAME, 
                        user_prompt=user_prompt, 
                        use_context=use_context
                    )
                    

                retriever = RETRIEVER_MAP[from_source or 'HPx']['hybrid_retriever']
                    
                # ----- Source extraction -----
                curated_prompt = user_prompt.split('\n<<retry_')[0]
                results_with_scores = retriever.get_relevant_documents(curated_prompt)

                for doc in results_with_scores:
                    logging.info(f"Document: {doc.metadata.get('source', 'Unknown Source')} | Score: {doc.metadata.get('score', 0)}")

                HIGH_THRESHOLD = 0.7
                MID_THRESHOLD = 0.61

                filtered_results = [
                    doc for doc, score in sorted(
                        [(doc, doc.metadata.get('score', 0)) for doc in results_with_scores
                        if doc.metadata.get('score', 0) >= HIGH_THRESHOLD],
                        key=lambda x: x[1],
                        reverse=True
                    )
                ]

                if len(filtered_results) == 0:
                    filtered_results = [
                        doc for doc, score in sorted(
                           [(doc, doc.metadata.get('score', 0)) for doc in results_with_scores
                           if doc.metadata.get('score', 0) >= MID_THRESHOLD],
                           key=lambda x: x[1], 
                           reverse=True
                    )
                ]

                # If sources don't enter the threshold then we check if the slopoe 
                # between the first two is steep enough to return the first one.
                if not filtered_results:
                    seen = set()
                    unique_results = []
                    
                    for doc in results_with_scores:
                        doc_source = doc.metadata.get("source", "Unknown Source")
                        score = doc.metadata.get("score", 0)
                        if doc_source not in seen:
                            seen.add(doc_source)
                            unique_results.append((doc, score))
                    sorted_results = sorted(unique_results, key=lambda x: x[1], reverse=True)

                    # if len(sorted_results) > 1 and (sorted_results[0][1] - sorted_results[1][1]) >= 0.025:
                    if len(sorted_results) > 1 and (sorted_results[0][1] - sorted_results[1][1]) >= 0.1:
                        filtered_results = [sorted_results[0][0]]
                
                # Out of scope questions
                if len(filtered_results) == 0:
                    oos_message = random.choice(OOS_MESSAGE)
                    for word in oos_message.split(' '):
                        time.sleep(0.08)
                        yield f"{word} "
                    return
                
                # Build stream chain
                question_answer_chain = create_stuff_documents_chain(LLM, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                chain = rag_chain.pick("answer")

                # Appending the stop callback to existing callbacks (so we don't overwrite the default stream logging)
                existing_callbacks = getattr(LLM, "callbacks", [])
                stop_handler = StopStreamHandler(stream_id, r_streams)
                LLM.callbacks = existing_callbacks + [stop_handler]

                answer = ''

                for token in chain.stream({"input": user_prompt}):
                    if r_streams.hget(f"stream:{stream_id}", "stop") == b"1":
                        logging.info(f"Stopping stream {stream_id} due to user request.")
                        break  # Exit the loop if stop signal is received
                    answer += token
                    yield token
                        
                logging.info("""
                             
                             """)
                

                sources = []
                sources_returned = 'no'
                unique_sources = set()
                for doc in filtered_results:
                    source = doc.metadata.get("source", "Unknown Source")
                    if source not in unique_sources and not "AUX_DOCS" in source:
                        unique_sources.add(source)
                        sources.append(source)

                if sources:
                    source_title = "Sources:" if len(sources) > 1 else "Source:"
                    yield f"\n <br/><br/><strong>{source_title}</strong>"
                    
                    for i, source in enumerate(sources[:4]):
                        unreleased_mark = ' (unreleased)' if 'UNRELEASED' in source else ''
                        time.sleep(0.15)  # Simulating streaming effect
                        sources_returned = 'yes'
                        yield f"\n- {source}{unreleased_mark}"
                logging.info('\n \n')
                logging.info(f"Sources returned: {sources_returned}")
                logging.info('\n \n')
                
                # if from_source == 'SNOK':
                #     snok_docs = RETRIEVER_MAP[from_source]['hybrid_retriever'].get_relevant_documents(user_prompt)
                #     if len(snok_docs):
                #         yield f"\n\n\nSnok Retriever:\n"
                #         for doc in snok_docs:
                #             source_title = doc.metadata.get('source','?').split(']')[0].replace('[','')
                #             yield f"\n\nSource={source_title} | Score={doc.metadata.get('score')}\n"


            # Chat with LLM only
            else:
                print(f"\n\n*** Using direct chat with LLM ***\n")
                prompt, memory = get_prompt_template(
                    model_name=MODEL_NAME, user_prompt=user_prompt, use_context=use_context
                )
                input_data = {"context": None, "question": user_prompt, "history": use_history}

                try:
                    print("\nAnswer: \n")
                    llm_chain = StreamingChain(LLM, prompt)
                    for token in llm_chain.stream(input_data=input_data):
                        if r_streams.hget(f"stream:{stream_id}", "stop") == b"1":
                            logging.info(f"Stopping stream {stream_id} due to user request.")
                            break  # Exit the loop if stop signal is received
                        yield token.encode("utf-8")
                    print("\n\n")
                except Exception as e:
                    logging.error(f"Error during streaming: {str(e)}")
                    yield "Error occurred during streaming".encode("utf-8")

        except KeyboardInterrupt:
            logging.info(f"Stream {stream_id} stopped by user")
            
        except Exception as e:
            error = str(e)
            logging.info(f">>> An error occurred while generating the reponse: {error}", exc_info=True)
            response.headers["error"] = error
            
            error_user_message = random.choice(TECH_ISSUE_LLM)
            ERROR_DOWN_URL = os.getenv('ERROR_DOWN_URL', 'nourl')
            error_message = f"""\n{error_user_message}
            \nPlease try again or start a new chat.
            \nVisit [Down@Volvo]({ERROR_DOWN_URL}) for more info."""
            
            yield '\n'
            for word in error_message.split(' '):
                time.sleep(0.1)
                yield f"{word} "
        
        finally:
            r_streams.delete(f"stream:{stream_id}")

    response = Response(generate_stream(), mimetype="text/plain")
    response.headers["Stream-ID"] = stream_id
    logging.info(f">>> Generated response for stream ID: {stream_id}")
    logging.info(">>> Exiting /api/prompt_route")
    return response


# ---- This is a testing route ----
@app.route("/api/prompt_route_test", methods=["GET", "POST"])
@require_token
def prompt_route_test():
    logging.info(">>> Entering /api/prompt_route_test")
    user_prompt = request.form.get("user_prompt")
    use_context = request.form.get("use_context")
    first_query = request.form.get("first_query")
    use_history = request.form.get("use_history")
    from_source = request.form.get("source")
    stream_id = int(request.form.get("stream_id", str(int(time.time() * 1000))))
    stop = request.form.get("stop", "false").lower() == "true"
    error = ''
    
    if not RETRIEVER_MAP[from_source]:
        print('\n')
        print(f'Category soruce: {from_source} not found. Falling back to HPx')
        print('\n')
        from_source = 'HPx'
        
    if stop:
        logging.info(f"Attempting to stop stream ID: {stream_id}")

        # Check if stream exists
        if not r_streams.exists(f"stream:{stream_id}"):
            return jsonify({"error": f"Stream {stream_id} not found"}), 404

        # Set stop flag (atomic)
        r_streams.hset(f"stream:{stream_id}", "stop", 1)

        logging.info(f"Stop signal sent for stream ID: {stream_id}")
        return jsonify({"message": f"Stream {stream_id} stop signal issued."}), 200

    if use_context is None:
        use_context = True
    else:
        use_context = use_context.lower() == "true"

    if not user_prompt:
        return "No user prompt received", 400

    if r_streams.exists(f"stream:{stream_id}"):
        return jsonify({"error": f"Stream {stream_id} is already in progress"}), 409

    r_streams.hset(f"stream:{stream_id}", mapping={"stop": 0})
    r_streams.expire(f"stream:{stream_id}", STREAM_TTL)

    def generate_stream():
        nonlocal error
        try:
            r_streams.expire(f"stream:{stream_id}", STREAM_TTL)
            stop_handler = StopStreamHandler(stream_id, r_streams)
            
            # Use documents context
            if use_context:
                print('\n \n')
                print('\n \n')
                print('*** Using TEST chat with CONTEXT ***')
                print(f'User prompt: {user_prompt}')
                print('\n \n')
                print(f"Source: {from_source}")
                print('\n \n')
                print(f"First query: {'yes' if first_query else 'no'}")
                print('\n \n')

                if from_source == 'SNOK':
                    prompt, memory = get_prompt_template(
                        system_prompt=SNOK_SYSTEM_PROMPT,
                        model_name=MODEL_NAME, 
                        user_prompt=user_prompt, 
                        use_context=use_context
                    )
                else:
                    prompt, memory = get_prompt_template(
                        model_name=MODEL_NAME, 
                        user_prompt=user_prompt, 
                        use_context=use_context
                    )
                    

                # Build stream chain
                retriever = RETRIEVER_MAP[from_source or 'HPx']['hybrid_retriever']
                question_answer_chain = create_stuff_documents_chain(LLM, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                chain = rag_chain.pick("answer")

                # Appending the stop callback to existing callbacks (so we don't overwrite the default stream logging)
                existing_callbacks = getattr(LLM, "callbacks", [])
                stop_handler = StopStreamHandler(stream_id, r_streams)
                LLM.callbacks = existing_callbacks + [stop_handler]

                answer = ''

                for token in chain.stream({"input": user_prompt}):
                    if r_streams.hget(f"stream:{stream_id}", "stop") == b"1":
                        logging.info(f"Stopping stream {stream_id} due to user request.")
                        break  # Exit the loop if stop signal is received
                    answer += token
                    yield token
                        
                logging.info('\n \n')
                logging.info('\n \n')
                    
                # ----- Source generation -----
                source_retriever = RETRIEVER_MAP[from_source or 'HPx']['semantic_retriever']
                results_with_scores = source_retriever.vectorstore.similarity_search_with_relevance_scores(answer, k=SEMANTIC_K_DOCS)

                for doc, score in results_with_scores:
                    logging.info(f"Document: {doc.metadata.get('source', 'Unknown Source')} | Score: {score}")

                HIGH_THRESHOLD = 0.71
                MID_THRESHOLD = 0.61

                filtered_results = sorted(
                    [(doc, score) for doc, score in results_with_scores if score >= HIGH_THRESHOLD],
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                if len(filtered_results) == 0:
                    filtered_results = sorted(
                        [(doc, score) for doc, score in results_with_scores if score >= MID_THRESHOLD],
                        key=lambda x: x[1], 
                        reverse=True
                    )

                retriever_results = [doc for doc, score in filtered_results]

                # If sources don't enter the threshold then we check if the slopoe 
                # between the first two is steep enough to return the first one.
                if not filtered_results:
                    seen = set()
                    unique_results = []
                    
                    for doc, score in results_with_scores:
                        doc_source = doc.metadata.get("source", "Unknown Source")
                        if doc_source not in seen:
                            seen.add(doc_source)
                            unique_results.append((doc, score))
                    sorted_results = sorted(unique_results, key=lambda x: x[1], reverse=True)

                    # if len(sorted_results) > 1 and (sorted_results[0][1] - sorted_results[1][1]) >= 0.025:
                    if len(sorted_results) > 1 and (sorted_results[0][1] - sorted_results[1][1]) >= 0.1:
                        retriever_results = [sorted_results[0][0]]

                sources = []
                sources_returned = 'no'
                unique_sources = set()
                for doc in retriever_results:
                    source = doc.metadata.get("source", "Unknown Source")
                    if source not in unique_sources and not "AUX_DOCS" in source:
                        unique_sources.add(source)
                        sources.append(source)

                if sources:
                    source_title = "Sources:" if len(sources) > 1 else "Source:"
                    yield f"\n <br/><br/><strong>{source_title}</strong>"
                    
                    for i, source in enumerate(sources[:4]):
                        unreleased_mark = ' (unreleased)' if 'UNRELEASED' in source else ''
                        time.sleep(0.15)  # Simulating streaming effect
                        sources_returned = 'yes'
                        yield f"\n- {source}{unreleased_mark}"
                logging.info('\n \n')
                logging.info(f"Sources returned: {sources_returned}")
                logging.info('\n \n')


            # Chat with LLM only
            else:
                print(f"\n\n*** Using direct chat with LLM ***\n")
                prompt, memory = get_prompt_template(
                    model_name=MODEL_NAME, user_prompt=user_prompt, use_context=use_context
                )
                input_data = {"context": None, "question": user_prompt, "history": use_history}

                try:
                    print("\nAnswer: \n")
                    llm_chain = StreamingChain(LLM, prompt)
                    for token in llm_chain.stream(input_data=input_data):
                        if r_streams.hget(f"stream:{stream_id}", "stop") == b"1":
                            logging.info(f"Stopping stream {stream_id} due to user request.")
                            break  # Exit the loop if stop signal is received
                        yield token.encode("utf-8")
                    print("\n\n")
                except Exception as e:
                    logging.error(f"Error during streaming: {str(e)}")
                    yield "Error occurred during streaming".encode("utf-8")

        except KeyboardInterrupt:
            logging.info(f"Stream {stream_id} stopped by user")
            
        except Exception as e:
            error = str(e)
            logging.info(f">>> An error occurred while generating the reponse: {error}", exc_info=True)
            response.headers["error"] = error
            
            error_user_message = random.choice(TECH_ISSUE_LLM)
            ERROR_DOWN_URL = os.getenv('ERROR_DOWN_URL', 'nourl')
            error_message = f"""\n{error_user_message}
            \nPlease try again or start a new chat.
            \nVisit [Down@Volvo]({ERROR_DOWN_URL}) for more info."""
            
            yield '\n'
            for word in error_message.split(' '):
                time.sleep(0.1)
                yield f"{word} "
        
        finally:
            r_streams.delete(f"stream:{stream_id}")

    response = Response(generate_stream(), mimetype="text/plain")
    response.headers["Stream-ID"] = stream_id
    logging.info(f">>> Generated response for stream ID: {stream_id}")
    logging.info(">>> Exiting /api/prompt_route")
    return response


# Other routes remain unchanged
@app.route("/api/delete_source", methods=["GET"])
@require_token
def delete_source_route():
    folder_name = "SOURCE_DOCUMENTS"

    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)

    os.makedirs(folder_name)

    return jsonify({"message": f"Folder '{folder_name}' successfully deleted and recreated."})


@app.route("/api/save_document", methods=["GET", "POST"])
@require_token
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
@require_token
def run_ingest_route():
    global DB
    global semantic_retriever
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
        semantic_retriever = DB.as_retriever()

        return "Script executed successfully: {}".format(result.stdout.decode("utf-8")), 200
    except Exception as e:
        return f"Error occurred: {str(e)}", 500


# ------ FEEDBACK -------

@app.route("/api/save_feedback", methods=["POST"])
@require_token
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
@require_admin_token
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
@require_admin_token
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
@require_admin_token
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


@app.route('/api/vectorstore_search', methods=['POST'])
@require_admin_token
def search_vectors():
    try:
        data = request.get_json()
        query = data['query']
        full_text = data['fulltext']
        k = int(data['k']) if data['k'] else 0
        source = data['source']
        logging.info(f"******** source: {source}")

        if not query:
            return jsonify({"error": "Query parameter is required"}), 400
                
        if full_text:
            special_words = get_uncommon_or_identifier_words(query)
            bm25_query = " ".join(quote_fts_token(w) for w in special_words)
            bm25_docs = search_fts(bm25_query, k or FULLTEXT_K_DOCS, db_path=f"{source}.db")
            bm25_docs_dicts = [doc.page_content for doc in bm25_docs]
            found_exact = any(query.lower() in doc_text.lower() for doc_text in bm25_docs_dicts)
            return jsonify({"query": query, "matches": bm25_docs_dicts, "exact": found_exact})
        
        search_retriever = RETRIEVER_MAP[source]['hybrid_retriever']
        results = search_retriever.get_relevant_documents(query)[:k]
        semantic_docs = [document_to_dict(d) for d in results]
        results_serializable = [doc["page_content"] for doc in semantic_docs]
        found_exact = any(query.lower() in doc_text.lower() for doc_text in results_serializable)
        
        # Return matches as JSON
        return jsonify({"query": query, "matches": results_serializable, "exact": found_exact, "results": semantic_docs})
    
    except Exception as e:
        error = str(e)
        logging.info(f">>> An error occurred while searching on vectorstore: {error}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    

@app.route("/api/save_analytics", methods=["POST"])
@require_token
def save_analytics():
    try:
        data = request.get_json()

        if "session_id" not in data:
            return jsonify({"error": "Missing required fields: session_id"}), 400

        # Set default values for optional fields
        token_count = data.get("token_count", 0)

        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analytics (
                    session_id, 
                    message_count, 
                    token_count, 
                    duration_seconds,
                    prompt,
                    messages
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                message_count = excluded.message_count,
                token_count = excluded.token_count,
                duration_seconds = excluded.duration_seconds,
                prompt = excluded.prompt,
                messages = excluded.messages
            """, (
                data["session_id"],
                data["message_count"],
                token_count,
                data["duration_seconds"],
                data["prompt"],
                json.dumps(data["messages"])
            ))
            conn.commit()

        return jsonify({"message": "Analytics saved successfully"}), 201

    except Exception as e:
        logging.info(f">>> An error occurred while saving analytics: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    
    
@app.route("/api/get_analytics", methods=["GET"])
@require_admin_token
@require_token
def get_analytics():
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM analytics")  # Fetch all columns
            analytic_list = [dict(row) for row in cursor.fetchall()]  # Convert rows to dicts

        return jsonify(analytic_list), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
@app.route("/api/delete_analytics", methods=["DELETE"])
@require_admin_token
def delete_analytics():
    try:
        data = request.get_json()
        session_id = data.get("session_id")

        if not session_id:
            return jsonify({"error": "session_id is required"}), 400

        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()

            cursor.execute("DELETE FROM analytics WHERE session_id = ?", (session_id,))
            conn.commit()

            if cursor.rowcount == 0:
                return jsonify({"error": "No record found with the given session_id"}), 404

        return jsonify({"message": "Analytic deleted successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/get_model_settings', methods=['GET'])
def get_model_settings():
    try:
        model_settings = {
            "model_name": MODEL_NAME,
            "db_date": DB_DATE.strftime("%a, %d %b %Y %H:%M:%S GMT") if hasattr(DB_DATE, "strftime") else str(DB_DATE),
            "device": DEVICE_TYPE,
            "ctx_size": CONTEXT_WINDOW_SIZE,
            "new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "r_penalty": R_PENALTY,
            "n_batch": N_BATCH,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "split_separators": SPLIT_SEPARATORS,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "retrieve_k": SEMANTIC_K_DOCS,
            "collection_meta": COLLECTION_METADATA,
            "model_basename": MODEL_BASENAME,
            "embeddings": EMBEDDING_MODEL_NAME
        }
        return jsonify(model_settings)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/api/get_app_version', methods=['GET'])
def get_app_version():
    try:
       with open('/chatbot/source/ui/src/constants/app.ts', 'r') as file:
        first_line = file.readline().strip()

        # Define a regular expression to match the version number (a float number)
        match = re.search(r"export const APP_VERSION = '([0-9]*\.?[0-9]+)'", first_line)
        
        if match:
            return jsonify({"app_version": match.group(1)})  # Return as JSON
        else:
            return jsonify({"error": "Version not found"}), 404
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
# ----------------------------
# DLT - GDPR COMPLIANCE ROTUES
# ----------------------------
from gdpr_scan_regex import process_logs, filter_and_mask_dlt
from convert_dlt_to_json import parse_original_logs
import uuid
from flask import send_file
import io


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB limit

# ----------------------------
# Upload DLT file
# ----------------------------
@app.route("/api/upload-dlt", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    logging.info('Uploading new DLT file...')
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Create a unique file ID
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1]
    saved_filename = f"{file_id}{ext}"
    file_path = os.path.join(UPLOAD_DIR, saved_filename)
    file.save(file_path)
    logging.info(f"Done uploading: {file.filename}")
    
    delete_file_later(file_path, delay_seconds=3600)

    return jsonify({"file_id": file_id}), 200

# ----------------------------
# Analyze file by ID
# ----------------------------
@app.route("/api/process-dlt/<file_id>", methods=["GET"])
def analyze_file(file_id):
    try:
        # Find the file by file_id (assume .dlt extension)
        matching_files = [f for f in os.listdir(UPLOAD_DIR) if f.startswith(file_id) and f.endswith('.dlt')]
        if not matching_files:
            return jsonify({"error": "File not found"}), 404

        file_path = os.path.join(UPLOAD_DIR, matching_files[0])
        json_path = file_path.replace(".dlt", ".json")
        logging.info(f"\nAnalyzing: {matching_files[0]}")

        # Convert DLT -> JSON
        parse_original_logs(input_file=file_path, output_file=json_path)

        # Load JSON and run GDPR scan
        with open(json_path, "r", encoding="utf-8") as f:
            logs = json.load(f)

        logging.info(f"Processing logs and creating flags...")
        processed_json = process_logs(logs)

        flagged_path = json_path.replace(".json", "_flagged.json")
        
        with open(flagged_path, "w", encoding="utf-8") as f:
            json.dump(processed_json, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Done analyzing: {matching_files[0]}")
        
        # Remove after 1 hour
        delete_file_later(json_path, delay_seconds=3600)

        return jsonify(processed_json), 200

    except Exception as e:
        logging.error("Error analyzing file %s: %s", file_id, str(e))
        return jsonify({"error": "Failed to process file"}), 500


# ----------------------------
# Process & mask DLT file by ID and return masked DLT
# ----------------------------
@app.route("/api/mask-dlt/<file_id>", methods=["GET"])
def mask_dlt_file(file_id):
    try:
        # Find JSON analysis for this file
        json_files = [f for f in os.listdir(UPLOAD_DIR) if f.startswith(file_id) and f.endswith("_flagged.json")]
        if not json_files:
            return jsonify({"error": "JSON file not found"}), 404

        json_path = os.path.join(UPLOAD_DIR, json_files[0])
        logging.info(f"Found file for maskin: {json_path}")

        # Load processed logs to determine flagged entries
        with open(json_path, "r", encoding="utf-8") as f:
            processed_logs = json.load(f)
        flagged_timestamps = {log["timestamp"] for log in processed_logs if log.get("findings")}
        logging.info(f"Found {len(flagged_timestamps)} flagged logs")
        
        # Original DLT path
        dlt_path = json_path.replace("_flagged.json", ".dlt")
        if not os.path.exists(dlt_path):
            return jsonify({"error": "Original DLT file missing"}), 500

        # Output masked DLT in memory
        logging.info(f"Generating new DLT file from flags...")
        output_buffer = NonClosingBytesIO()

        # Use imported filter_and_mask_dlt function
        filter_and_mask_dlt(
            input_file=dlt_path,
            output_file=output_buffer,  # can pass BytesIO object
            flagged_timestamps=flagged_timestamps
        )

        logging.info(f"Done generating masked DLT.")
        # Return masked DLT
        return send_file(
            io.BytesIO(output_buffer.getvalue()),
            mimetype="application/octet-stream",
            download_name=f"{file_id}_masked.dlt",
            as_attachment=True
        )

    except Exception as e:
        logging.error(f"Error masking DLT {file_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to mask file"}), 500



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
