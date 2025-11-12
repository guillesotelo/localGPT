import os
import csv
from datetime import datetime
import re
import logging
from constants import EMBEDDING_MODEL_NAME, MODEL_ID
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import AutoModel, AutoTokenizer
import json

from langchain.schema import BaseRetriever, Document
from typing import List, Any


def log_to_csv(question, answer):

    log_dir, log_file = "local_chat_history", "qa_log.csv"
    # Ensure log directory exists, create if not
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Construct the full file path
    log_path = os.path.join(log_dir, log_file)

    # Check if file exists, if not create and write headers
    if not os.path.isfile(log_path):
        with open(log_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "question", "answer"])

    # Append the log entry
    with open(log_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, question, answer])


def get_embeddings(device_type="cuda"):
    if "instructor" in EMBEDDING_MODEL_NAME:
        return HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type},
            embed_instruction="Represent the document for retrieval:",
            query_instruction="Represent the question for retrieving supporting documents:",
        )

    elif "bge" in EMBEDDING_MODEL_NAME:
        encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
        return HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type},
            encode_kwargs=encode_kwargs
            # Removing this for BGE embedding model
            # query_instruction="Represent this sentence for searching relevant passages:",
        )
    
    elif "jina" in EMBEDDING_MODEL_NAME or "gte" in EMBEDDING_MODEL_NAME:
        # Use HuggingFaceEmbeddings for Jina model, adding `trust_remote_code`
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={
                "device": device_type,
                "trust_remote_code": True  # Allow custom code execution
            },
        )
    
    else:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type},
        )


def parse_table(table_text, format_type="key-value"):
    """
    Converts a Markdown-style table into a structured plain-text format.
    
    format_type:
        - "key-value": Converts each row into key-value pairs using headers.
        - "tabular-list": Converts the table into a readable bulleted list.
    
    Returns formatted table text.
    """
    lines = table_text.strip().split("\n")

    # Extract headers and rows
    headers = [h.strip() for h in lines[0].split("|")[1:-1]]  # Remove empty first and last splits
    data_rows = [line.split("|")[1:-1] for line in lines[2:] if "|" in line]

    formatted_table = ""

    if format_type == "key-value":
        # Format as "Column: Value" per row
        for row in data_rows:
            row_values = [v.strip() for v in row]
            formatted_table += "\n".join([f"{headers[i]}: {row_values[i]}" for i in range(len(headers))])
            formatted_table += "\n\n"  # Space between rows for clarity

    elif format_type == "tabular-list":
        # Format as a bulleted list
        formatted_table += "\n".join([f"- {', '.join([headers[i] + ': ' + row[i].strip() for i in range(len(headers))])}" for row in data_rows])
    
    return formatted_table.strip()

def process_document_with_tables(input_text, table_format="key-value", file_name=''):
    """
    Detects and converts tables into plain text for easier model processing.
    """

    # We return text as-is if no tables detected
    if "|" not in input_text:  # No table detected
        return input_text
    
    if file_name:
        logging.info(f"\nProcessing table from: {file_name}\n")
    
    table_pattern = re.compile(r"(\|.*?\|\n(?:\|[-:]+\|.*\n)*)", re.MULTILINE)
    tables = table_pattern.findall(input_text)

    processed_text = input_text

    for table in tables:
        formatted_table = parse_table(table, format_type=table_format)
        processed_text = processed_text.replace(table, formatted_table)

    return processed_text

# # Example Document with a Table
# document_text = """Here is some introductory text.

# | Name  | Age | Occupation |
# |-------|-----|-----------|
# | Alice | 30  | Engineer  |
# | Bob   | 25  | Designer  |

# The text continues after the table."""

# # Process the document to convert tables into readable text
# processed_document = process_document(document_text, table_format="key-value")

# # Print the final result
# print("Processed Document:\n")
# print(processed_document)


def document_to_dict(doc):
    result = {}
    for k, v in doc.__dict__.items():
        try:
            json.dumps(v)  # test if value is JSON-serializable
            result[k] = v
        except TypeError:
            result[k] = str(v)  # fallback: convert non-serializable to string
    return result


class ExactChromaRetriever(BaseRetriever):
    db: Any
    embeddings: Any
    k: int = 4

    def get_relevant_documents(self, query: str) -> list[Document]:
        query_embedding = self.embeddings.embed_query(query)
        results = self.db._collection.query(
            query_embeddings=[query_embedding],
            n_results=self.k
        )

        documents = []
        for i, item in enumerate(results["documents"][0]):
            score = results.get("distances", [[]])[0][i] if "distances" in results else None
            doc_metadata = {"score": score} if score is not None else {}
            documents.append(Document(page_content=item, metadata=doc_metadata))
        return documents


def get_collection_size(db):
    try:
        collection = db._collection.get()
        logging.info(f"Collection length: {len(collection['documents'])}")
        logging.info(f"Document lengths: {[len(doc) for doc in collection['documents']]}")
    except Exception as e:
        logging.info(f"Error getting collection size: {str(e)}", exc_info=True)
        
        