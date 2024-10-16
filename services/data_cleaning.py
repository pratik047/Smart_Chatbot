from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def clean_data(filename: str):
    # Load and split PDF documents
    loader = PyPDFLoader(filename)
    data = loader.load()

    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    # Generate document texts, metadatas, and unique IDs
    content = [doc.page_content for doc in docs]  # Extract text from each document
    ids = [f"doc_{i}" for i in range(len(content))]  # Assign unique IDs

    return (docs, ids)
