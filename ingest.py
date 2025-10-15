import logging
import os
import shutil
import psutil
import time
import stat
import pickle

import click
import torch
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from chromadb import PersistentClient

import nltk
from utils import get_embeddings, process_document_with_tables
from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    PERSIST_DIRECTORY_SNOK,
    SOURCE_DIRECTORY,
    SOURCE_DIRECTORY_SNOK,
    AUX_DOCS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_METADATA,
    SPLIT_SEPARATORS
)
from update_datetime import update_auxiliary_data_file

SERVER_URL = os.getenv('SERVER_URL', '')

nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")

os.environ["CURL_CA_BUNDLE"] = ""
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def file_log(logentry):
    with open("ingestion.log", "a") as f:
        f.write(logentry)
    print(logentry)


def remove_readonly(func, path, _):
    """Clear read-only flag and retry deletion."""
    os.chmod(path, 0o777)
    func(path)


def force_delete_chroma_db(db_path, collection_name="langchain", max_retries=3, retry_delay=2):
    """Forcefully delete the ChromaDB collection and database directory."""
    if os.path.exists(db_path):
        logging.info(f"Attempting to delete ChromaDB at: {db_path}")

        # Step 1: Connect to ChromaDB and delete the collection
        try:
            client = PersistentClient(path=db_path)
            client.delete_collection(collection_name)
            logging.info(f"Deleted ChromaDB collection: {collection_name}")
        except Exception as e:
            logging.warning(f"Could not delete collection {collection_name}: {e}")

        # Step 2: Kill any process using ChromaDB
        for proc in psutil.process_iter(['pid', 'name', 'open_files']):
            try:
                if proc.info['open_files']:
                    for file in proc.info['open_files']:
                        if db_path in file.path:
                            logging.info(f"Killing process {proc.info['name']} (PID: {proc.info['pid']})")
                            proc.terminate()
                            proc.wait(timeout=3)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Step 3: Retry deletion with permissions fix
        for attempt in range(max_retries):
            try:
                shutil.rmtree(db_path, onerror=remove_readonly)
                if not os.path.exists(db_path):
                    logging.info(f"Successfully deleted DB folder: {db_path}")
                    return
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(retry_delay)

        logging.error(f"Failed to delete DB folder after {max_retries} attempts. Delete it manually.")
    else:
        logging.info(f"ChromaDB folder {db_path} does not exist.")


def load_documents_from_directory(directory, documents):
    """Load and append documents from the given directory to the list."""
    total_textfiles = 0
    for root, _, files in os.walk(directory):
        for file_name in sorted(files):
            ext = os.path.splitext(file_name)[1]
            if ext not in (".txt", ".md", ".rst"):
                continue

            total_textfiles += 1
            # print(f"Importing: {file_name}")
            source_file_path = os.path.join(root, file_name)
            try:
                file_log(f"{source_file_path} loaded.")
                loader_class = DOCUMENT_MAP.get(ext if ext in DOCUMENT_MAP else '.txt')
                loader = loader_class(source_file_path)
                document = loader.load()[0]

                # metadata
                if '§' in file_name:
                    spliturl = file_name[4:].replace('¤', '/').split('§')
                    url_ext = '.md' if ext == '.md' else '.html'
                    url = f"[{spliturl[0]}]({SERVER_URL}{spliturl[1].replace(ext, url_ext)})"
                    document.metadata["source"] = url

                if len(document.page_content) > 300:
                    documents.append(document)

            except Exception as ex:
                file_log(f"{source_file_path} loading error: \n{ex}")

    return total_textfiles


def ingest_environment(env_name, source_directory, persist_directory, embeddings, device_type):
    """Reusable ingestion for a single environment (eg: HPx or SNOK)."""
    logging.info(f"[{env_name}] Deleting existing DB at {persist_directory}")
    force_delete_chroma_db(persist_directory)

    logging.info(f"[{env_name}] Loading documents from {source_directory} and {AUX_DOCS}...")
    documents = []
    total_textfile_count = 0

    total_textfile_count += load_documents_from_directory(source_directory, documents)
    total_textfile_count += load_documents_from_directory(AUX_DOCS, documents)

    if len(documents) == 0:
        logging.warning(f"[{env_name}] No documents found. Skipping ingestion.")
        return

    # Split documents into text chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=SPLIT_SEPARATORS, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)

    # Save docs for bm25 retrieval (optional, only for primary env usually)
    bm25_path = f"bm25_docs_{env_name.lower()}.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(texts, f)

    logging.info(f"[{env_name}] Loaded {len(documents)}/{total_textfile_count} documents")
    logging.info(f"[{env_name}] Split into {len(texts)} chunks")

    # Create the Chroma DB from documents
    Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=persist_directory,
        client_settings=CHROMA_SETTINGS,
        collection_metadata=COLLECTION_METADATA
    )

    logging.info(f"[{env_name}] Successfully ingested into {persist_directory}")


@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu", "cuda", "ipu", "xpu", "mkldnn", "opengl", "opencl", "ideep", "hip",
            "ve", "fpga", "ort", "xla", "lazy", "vulkan", "mps", "meta", "hpu", "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
def main(device_type):
    # Update date time in auxiliary data
    update_auxiliary_data_file()

    # Generate embeddings once, reuse for both environments
    embeddings = get_embeddings(device_type)

    # Ingest HPx environment
    ingest_environment(
        env_name="HPx",
        source_directory=SOURCE_DIRECTORY,
        persist_directory=PERSIST_DIRECTORY,
        embeddings=embeddings,
        device_type=device_type
    )
    logging.info("HPx Chroma DB ingested successfully.")

    # Ingest SNOK environment
    ingest_environment(
        env_name="SNOK",
        source_directory=SOURCE_DIRECTORY_SNOK,
        persist_directory=PERSIST_DIRECTORY_SNOK,
        embeddings=embeddings,
        device_type=device_type
    )

    logging.info("SNOK Chroma DB ingested successfully.")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("debug.log", mode="w"),
        ],
    )

    main()
