import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from .constants import LOGGER_KEY

logger = logging.getLogger(LOGGER_KEY)


def clean_data(filename: str):
    # Load and split PDF documents
    loader = PyPDFLoader(filename)
    data = loader.load()

    logger.info(f"PDF {filename} loaded...")

    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    # Generate document texts, metadatas, and unique IDs
    content = [doc.page_content for doc in docs]  # Extract text from each document
    ids = [f"doc_{i}" for i in range(len(content))]  # Assign unique IDs

    return (docs, ids)
