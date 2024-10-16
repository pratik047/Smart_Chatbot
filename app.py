import logging

import streamlit as st
from dotenv import load_dotenv

from services import clean_data, invoke, train_model

logger = logging.getLogger(__name__)


def main():
    # Clean
    docs, meta_data, ids = clean_data("media/research.pdf")
    logger.info("Data cleaning done...")

    # Updating the variable with global scope.
    global prompt, llm, retriever
    prompt, llm, retriever = train_model(docs, ids)
    logger.info("model training done...")

    logger.info("Data cleaning successful.")


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # initialize LLM with RAG
    main()

    # Streamlit application title
    st.title(" RAG Application built on Gemini Model ")

    st.write(f"Loaded {10} document chunks.")

    st.write("Embeddings generated successfully for sample documents.")

    # Take user input through Streamlit chat input
    query = st.chat_input("Ask something: ")

    if query is not None:
        st.write(f"User query: {query}")
        # Invoke the response.
        answer = invoke(query, llm, prompt, retriever)
        # Display the response
        st.write(answer)
    else:
        st.info("Please input a query.")
