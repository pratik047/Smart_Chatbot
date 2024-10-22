import logging

import streamlit as st
from dotenv import load_dotenv

from services import LOGGER_KEY, clean_data, initialize_logger, invoke, train_model

logger = logging.getLogger(LOGGER_KEY)


def main() -> int:
    # Clean
    docs, ids = clean_data("media/research.pdf")
    logger.info("Data cleaning done...")

    # Updating the variable with global scope.
    global prompt, llm, retriever
    prompt, llm, retriever = train_model(docs, ids)
    logger.info("model training done...")

    logger.info("Data cleaning successful.")

    return len(docs)


if __name__ == "__main__":
    # Logger initialization.
    initialize_logger()

    logger.info("Application loaded")
    # Load environment variables
    load_dotenv()

    # initialize LLM with RAG
    chunk_size = main()

    # Streamlit application title
    st.title("Smart chatbot using RAG!")

    st.info(f"Loaded {chunk_size} document chunks.")

    # Take user input through Streamlit chat input
    query = st.chat_input("Ask something: ")

    if query is not None:
        st.write(f"User query: {query}")
        # Invoke the response.
        answer = invoke(query, llm, prompt, retriever)
        # Display the response
        st.write(answer)
