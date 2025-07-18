import os
import argparse
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from utils import get_embeddings
from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    PERSIST_DIRECTORY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SPLIT_SEPARATORS
)
SERVER_URL = os.getenv('SERVER_URL', '')

def ingest_single_file(file_path, device_type='cuda'):
    # Load the existing Chroma DB (if it exists)
    embeddings = get_embeddings(device_type)
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS
    )

    # Extract file extension
    file_extension = os.path.splitext(file_path)[1]
    if file_extension != ".txt":
        logging.warning(f"Skipping unsupported file type: {file_path}")
        return

    print(f"Ingesting file: {file_path}")
    try:
        loader_class = DOCUMENT_MAP.get('.txt')
        loader = loader_class(file_path)
        document = loader.load()[0]

        # Parse metadata
        file_name = os.path.basename(file_path)
        if '§' in file_name:
            spliturl = file_name[4:].replace('¤', '/').split('§')
            url = f"[{spliturl[0]}]({SERVER_URL}{spliturl[1].replace('.txt', '.html')})"
            document.metadata["source"] = url

        # Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(separators=SPLIT_SEPARATORS, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        texts = text_splitter.split_documents([document])

        logging.info(f"Split into {len(texts)} chunks")

        # Add new documents to the existing DB
        db.add_documents(texts)

        # Persist the updated database
        db.persist()
        logging.info(f"Successfully added {len(texts)} chunks to the database.")

    except Exception as ex:
        logging.error(f"Error ingesting {file_path}: {ex}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a single file into ChromaDB.")
    parser.add_argument("file_path", type=str, help="Path to the file to be ingested")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"],
                        help="Device to use for embeddings (default: cuda)")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ingest_single_file(args.file_path, args.device)