import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import shutil
import psutil
import time
import stat

import click
import torch
from langchain.docstore.document import Document
# from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from utils import get_embeddings
from chromadb import PersistentClient
from utils import process_document_with_tables

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
    AUX_DOCS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_METADATA,
    SPLIT_SEPARATORS
)

import nltk

SERVER_URL = os.getenv('SERVER_URL', '')

nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")

os.environ["CURL_CA_BUNDLE"] = ""
# os.environ["CURL_CA_BUNDLE"] = "/chatbot/certs/huggingface.crt"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def file_log(logentry):
    file1 = open("file_ingest.log", "a")
    file1.write(logentry + "\n")
    file1.close()
    print(logentry + "\n")


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
        logging.info("ChromaDB folder does not exist.")

@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
def main(device_type):
    # Delete DB folder
    force_delete_chroma_db(PERSIST_DIRECTORY)

    # Load documents and split in chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY} and {AUX_DOCS}...")

    # List to hold documents
    documents = []

    # Function to load documents from a given directory
    def load_documents_from_directory(directory):
        for root, _, files in os.walk(directory):
            for file_name in sorted(files):
                file_extension = os.path.splitext(file_name)[1]
                if file_extension == '.txt':
                    print(f"Importing: {file_name}")
                    source_file_path = os.path.join(root, file_name)
                    try:
                        file_log(f"{source_file_path} loaded.")
                        loader_class = DOCUMENT_MAP.get('.txt')
                        loader = loader_class(source_file_path)
                        document = loader.load()[0]

                        # Parse tables (if needed)
                        # processed_text = process_document_with_tables(document.page_content, table_format="key-value", file_name=file_name)
                        # document.page_content = processed_text

                        # metadata
                        if '§' in file_name:
                            spliturl = file_name[4:].replace('¤', '/').split('§')
                            url = f"[{spliturl[0]}]({SERVER_URL}{spliturl[1].replace('.txt', '.html')})"
                            document.metadata["source"] = url

                        documents.append(document)

                    except Exception as ex:
                        file_log(f"{source_file_path} loading error: \n{ex}")

    # Load documents from both source directories
    load_documents_from_directory(SOURCE_DIRECTORY)
    load_documents_from_directory(AUX_DOCS)

    if len(documents) == 0:
        exit()

    # Split documents into text chunks
    text_splitter = RecursiveCharacterTextSplitter(separators=SPLIT_SEPARATORS, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)

    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY} and {AUX_DOCS}")
    logging.info(f"Split into {len(texts)} chunks of text")

    # Generate embeddings
    embeddings = get_embeddings(device_type)

    # Create the Chroma DB from documents
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
        collection_metadata=COLLECTION_METADATA
    )

    logging.info(f"*** Successfully loaded embeddings from {EMBEDDING_MODEL_NAME} ***")



if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        level=logging.DEBUG,  # Set to DEBUG for very verbose logging
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler("debug.log", mode="w"),  # Log to file
        ],
    )

    main()
