import argparse
import os
import shutil
import psutil
import time
import sqlite3
import json

import click
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from chromadb import PersistentClient

import nltk
from utils import get_embeddings
from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
    AUX_DOCS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_METADATA,
    SPLIT_SEPARATORS,
    CATEGORY_MAP
)
from update_datetime import update_auxiliary_data_file

SERVER_URL = os.getenv('SERVER_URL', '')

def create_fts_table(db_path="fts_index.db"):
    # Remove existing DB file
    if os.path.exists(db_path):
        os.remove(db_path)
        
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE VIRTUAL TABLE docs USING fts5(
            page_content,          -- searchable text
            source UNINDEXED,      -- metadata: file or doc name
            chunk_id UNINDEXED,    -- metadata: chunk index
            metadata_json UNINDEXED
        );
    """)
    conn.commit()
    conn.close()
    

def insert_documents_for_fts(docs, db_path="fts_index.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    for doc in docs:
        meta = doc.metadata or {}
        cur.execute("""
            INSERT INTO docs (page_content, source, chunk_id, metadata_json)
            VALUES (?, ?, ?, ?)
        """, (
            doc.page_content,
            meta.get("source", ""),
            meta.get("chunk_id", ""),
            json.dumps(meta),
        ))

    conn.commit()
    conn.close()


def load_documents_from_directory(directory, category_name, documents):
    """
    Load documents from directory, but only append those matching the category.
    HPx category receives ALL files.
    """
    filters = CATEGORY_MAP.get(category_name, [])
    total_textfiles = 0

    for root, _, files in os.walk(directory):
        for file_name in sorted(files):
            ext = os.path.splitext(file_name)[1]
            if ext not in (".txt", ".md", ".rst"):
                continue

            total_textfiles += 1

            source_file_path = os.path.join(root, file_name)

            try:
                print(f"[{category_name}] {source_file_path} loaded.")
                loader_class = DOCUMENT_MAP.get(ext if ext in DOCUMENT_MAP else '.txt')
                document = loader_class(source_file_path).load()[0]

                # Metadata URL mapping
                if '§' in file_name:
                    spliturl = file_name[4:].replace('¤', '/').split('§')
                    url_ext = '.md' if ext == '.md' else '.html'
                    url = f"[{spliturl[0]}]({SERVER_URL}{spliturl[1].replace(ext, url_ext)})"
                    document.metadata["source"] = url

                if len(document.page_content) > 50:
                    documents.append(document)

            except Exception as ex:
                print(f"{source_file_path} loading error: \n{ex}")

    return total_textfiles

def ingest_category(category: str, source_path: str, persist_root: str = "DB"):
   
    embeddings = get_embeddings("cuda")

    # Directory where THIS category’s DB will be stored
    category_dir = os.path.join(persist_root, category.lower())

    # Create DB folder if missing — DO NOT DELETE OTHERS
    os.makedirs(category_dir, exist_ok=True)

    documents = []
    total_textfiles = load_documents_from_directory(
      directory=source_path,
        category_name=category,
        documents=documents
    )
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        separators=SPLIT_SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)

    create_fts_table(db_path=f"{category}.db")
    insert_documents_for_fts(texts, db_path=f"{category}.db")

    print(f"Loaded {total_textfiles}/{len(documents)} new docs")

    # Initialize / load Chroma for this category only
    db = Chroma(
        embedding_function=embeddings,
        persist_directory=category_dir,
        client_settings=CHROMA_SETTINGS,
        collection_metadata=COLLECTION_METADATA
    )

    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        db.add_documents(batch)
    db.persist()
    
    print(f"Signle ingestion complete for: {category}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, help="e.g. SNOK, HPXA, ZUUL")
    parser.add_argument("--path", required=True, help="Directory containing new files")

    args = parser.parse_args()
    ingest_category(args.category, args.path)
