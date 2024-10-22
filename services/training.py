import logging

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from .constants import LOGGER_KEY

logger = logging.getLogger(LOGGER_KEY)


def train_model(docs, ids):
    # Test the embedding model to ensure it's working
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    logger.info("Embeddings generated successfully for sample documents.")

    # Create the vectorstore with embeddings
    store = Chroma.from_documents(documents=docs, embedding=embedding, ids=ids)
    logger.info("vector store done.")

    # Set up retriever with vector store
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    logger.info("retriever configuration done.")

    # Set up LLM (Gemini 1.5 Pro)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None
    )

    logger.info("LLM setup done.")

    # Define the system prompt for the assistant
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    # Create a ChatPromptTemplate for system and user input
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    return (prompt, llm, retriever)


def invoke(query, llm, prompt, retriever):
    logger.info(f"user query: {query}")
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Invoke the chain with the input query
    response = rag_chain.invoke({"input": query})

    # Display the response
    answer = response["answer"]
    logger.info(f"LLM answer: {answer}")

    return answer
