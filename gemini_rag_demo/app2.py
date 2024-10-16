import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

# Streamlit application title
st.title(" RAG Application built on Gemini Model ")

# Load and split PDF documents
loader = PyPDFLoader("Reasearch.pdf")
data = loader.load()

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Generate document texts, metadatas, and unique IDs
texts = [doc.page_content for doc in docs]  # Extract text from each document
metadatas = [doc.metadata for doc in docs]  # Extract metadata (if any)
ids = [f"doc_{i}" for i in range(len(texts))]  # Assign unique IDs

# Check if docs have content before proceeding
if len(texts) == 0:
    st.error("No texts found in the documents to add to Chroma.")
else:
    st.write(f"Loaded {len(docs)} document chunks.")

    try:
        # Test the embedding model to ensure it's working
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        test_embeddings = embedding.embed_documents(
            texts[:5]
        )  # Test embedding with first 5 documents
        st.write("Embeddings generated successfully for sample documents.")

        # Create the vectorstore with embeddings
        vectorstore = Chroma.from_documents(
            documents=docs, embedding=embedding, ids=ids
        )

        # Set up retriever with vector store
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 10}
        )

        # Set up LLM (Gemini 1.5 Pro)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None
        )

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

        # Take user input through Streamlit chat input
        query = st.chat_input("Ask something: ")

        # If a query is provided, process it
        if query:
            st.write(f"User query: {query}")
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            # Invoke the chain with the input query
            response = rag_chain.invoke({"input": query})

            # Display the response
            st.write(response["answer"])

        else:
            st.info("Please input a query.")

    except Exception as e:
        st.error(f"Error during RAG process: {e}")
