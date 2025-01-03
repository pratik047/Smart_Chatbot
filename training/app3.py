import logging
import spacy
import streamlit as st
from dotenv import load_dotenv
import networkx as nx
from pyvis.network import Network
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

import tempfile

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG_Application")

# Load environment variables
load_dotenv()

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")
logger.info("SpaCy model loaded successfully.")

# Streamlit application title
st.title("RAG-GPT")

# Initialize session state for graph visibility
if "show_graph" not in st.session_state:
    st.session_state.show_graph = False

# File upload for user input
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Load PDF and process documents
        loader = PyPDFLoader(temp_file_path)
        data = loader.load()
        logger.info(f"Loaded {len(data)} pages from PDF.")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        docs = text_splitter.split_documents(data)
        logger.info(f"Split documents into {len(docs)} chunks.")

        # Create embeddings and vector store
        persist_dir = "./chroma_data"  # Set persist directory

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        try:
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=persist_dir,
            )
            retriever = vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 10}
            )
        except Exception as e:
            logger.error(f"Error initializing the vector store: {e}")
            st.error(f"An error occurred while initializing the vector store: {e}")
            st.stop()

        # Load LLM (Gemini Model)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None
        )

        # Function to build knowledge graph
        def build_knowledge_graph(docs):
            graph = nx.DiGraph()

            for doc in docs:
                doc_nlp = nlp(doc.page_content)
                for ent in doc_nlp.ents:
                    graph.add_node(
                        ent.text,
                        label=ent.label_,
                        color="#1f78b4" if ent.label_ == "PERSON" else "#33a02c",
                    )

                for sent in doc_nlp.sents:
                    for token in sent:
                        if token.dep_ == "ROOT" and token.head.pos_ == "VERB":
                            subject = [w for w in token.lefts if w.dep_ == "nsubj"]
                            objects = [w for w in token.rights if w.dep_ == "dobj"]
                            if subject and objects:
                                graph.add_edge(
                                    subject[0].text,
                                    objects[0].text,
                                    relation=token.text,
                                    color="#ff7f00",
                                )
            return graph

        # Function to render knowledge graph
        def render_knowledge_graph(graph):
            net = Network(height="500px", width="100%", directed=True)

            for node, data in graph.nodes(data=True):
                net.add_node(
                    node,
                    title=f"{node} ({data.get('label', 'Unknown')})",
                    color=data.get("color", "#999999"),
                )

            for src, dest, data in graph.edges(data=True):
                relation = data.get("relation", "")
                net.add_edge(
                    src, dest, title=relation, color=data.get("color", "#999999")
                )

            return net.generate_html("knowledge_graph.html")

        # Button to toggle Knowledge Graph display
        if st.button("Knowledge Graph"):
            st.session_state.show_graph = not st.session_state.show_graph

        if st.session_state.show_graph:
            knowledge_graph = build_knowledge_graph(docs)
            graph_html = render_knowledge_graph(knowledge_graph)
            st.subheader("Knowledge Graph")
            st.components.v1.html(graph_html, height=550)

        # Handle user query
        query = st.chat_input("Ask something:")
        if query:
            st.write(f"User query: {query}")

            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise.\n\n{context}"
            )

            prompt = ChatPromptTemplate.from_messages(
                [("system", system_prompt), ("human", "{input}")]
            )
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            try:
                response = rag_chain.invoke({"input": query})
                st.write(response["answer"])
            except Exception as e:
                st.error(f"Error during RAG process: {e}")

    except Exception as e:
        logger.error(f"Error during PDF processing: {e}")
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a PDF file to proceed.")
