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
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from utils import get_embeddings

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)

import nltk

nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")

os.environ["CURL_CA_BUNDLE"] = ""


def file_log(logentry):
    file1 = open("file_ingest.log", "a")
    file1.write(logentry + "\n")
    file1.close()
    print(logentry + "\n")


def remove_readonly(func, path, _):
    """Clear read-only permissions and retry deletion."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def force_delete_chroma_db(db_path, max_retries=3, retry_delay=2):
    if os.path.exists(db_path):
        logging.info(f"Attempting to delete ChromaDB at: {db_path}")

        # Kill any process using ChromaDB
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
        
        # Retry deletion with permissions fix
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
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}...")

    # Loads all documents from the source documents directory, including nested folders
    documents = []
    for root, _, files in os.walk(SOURCE_DIRECTORY):
        for file_name in sorted(files):
            file_extension = os.path.splitext(file_name)[1]
            if file_extension == '.txt':
                print("Importing: " + file_name)
                source_file_path = os.path.join(root, file_name)
                try:
                    file_log(source_file_path + " loaded.")
                    loader_class = DOCUMENT_MAP.get('.txt')
                    loader = loader_class(source_file_path)
                    document = loader.load()[0]

                    # metadata
                    if '§' in file_name:
                        spliturl = file_name[4:].replace('¤','/').split('§') 
                        url = '[' + spliturl[0] + '](' + 'https://hpdevp.server-name.net/' + spliturl[1].replace('.txt','.html)')
                        document.metadata["source"] = url
                    documents.append(document)

                    if len(documents) == 0:
                        exit()

                except Exception as ex:
                    file_log("%s loading error: \n%s" % (source_file_path, ex))

    #text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n\n","\n\n","\n"," ",".",",",""],chunk_size=512, chunk_overlap=128)
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". "],chunk_size=512, chunk_overlap=256)
    texts = text_splitter.split_documents(documents)
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    embeddings = get_embeddings(device_type)

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
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
