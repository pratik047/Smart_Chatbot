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

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "show_graph" not in st.session_state:
    st.session_state.show_graph = False
if "knowledge_graph" not in st.session_state:
    st.session_state.knowledge_graph = None
if "user_data" not in st.session_state:
    st.session_state.user_data = {}


# Function to handle user login
def login(username, password):
    if (
        username in st.session_state.user_data
        and st.session_state.user_data[username] == password
    ):
        st.session_state.logged_in = True
        return True
    return False


# Register User
def register(username, password):
    if username in st.session_state.user_data:
        return False  # Username already exists
    st.session_state.user_data[username] = password
    return True


# Login Page
def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if login(username, password):
            st.success("Logged in successfully!")
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid username or password")

    if st.button("Register"):
        if register(username, password):
            st.success("Registration successful!")
        else:
            st.error("Username already exists")


# Knowledge Graph Functions
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
                if token.dep_ in ["amod", "compound", "prep"]:
                    graph.add_edge(
                        token.head.text,
                        token.text,
                        relation=token.dep_,
                        color="#e31a1c",
                    )
    return graph


def render_knowledge_graph(graph):
    net = Network(
        height="600px",
        width="100%",
        directed=True,
        bgcolor="#222222",
        font_color="white",
    )

    for node, data in graph.nodes(data=True):
        net.add_node(
            node,
            title=f"{node} ({data.get('label', 'Unknown')})",
            color=data.get("color", "#999999"),
            shape="dot",
        )

    for src, dest, data in graph.edges(data=True):
        net.add_edge(
            src,
            dest,
            title=data.get("relation", ""),
            color=data.get("color", "#999999"),
        )

    return net.generate_html("knowledge_graph.html")


# File Upload and Chatbot Page
def upload_and_chatbot_page():
    st.title("Upload PDF and Chatbot")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        data = loader.load()
        logger.info(f"Loaded {len(data)} pages from PDF.")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        docs = text_splitter.split_documents(data)
        logger.info(f"Split documents into {len(docs)} chunks.")

        # Create embeddings and vector store
        persist_dir = "./chroma_data"
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        try:
            vectorstore = Chroma.from_documents(
                documents=docs, embedding=embeddings, persist_directory=persist_dir
            )
            retriever = vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 10}
            )
        except Exception as e:
            logger.error(f"Error initializing the vector store: {e}")
            st.error(f"An error occurred while initializing the vector store: {e}")
            return

        # Load LLM (Gemini Model)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

        # Toggle Knowledge Graph
        if st.button("Knowledge Graph"):
            st.session_state.show_graph = not st.session_state.show_graph
            if st.session_state.show_graph:
                st.session_state.knowledge_graph = build_knowledge_graph(docs)

        if st.session_state.show_graph and st.session_state.knowledge_graph:
            graph_html = render_knowledge_graph(st.session_state.knowledge_graph)
            st.subheader("Knowledge Graph")
            st.components.v1.html(graph_html, height=650)

        # User Query Handling
        query = st.chat_input("Ask something:")
        if query:
            st.write(f"User query: {query}")

            if (
                st.session_state.knowledge_graph
                and query in st.session_state.knowledge_graph.nodes
            ):
                st.write(
                    f"Knowledge Graph answer: {st.session_state.knowledge_graph.nodes[query]}"
                )
            else:
                system_prompt = (
                    "You are an assistant for question-answering tasks. "
                    "Use the retrieved context to answer the question. "
                    "If you don't know the answer, say so. Keep the answer concise.\n\n{context}"
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


# Main App Flow
if st.session_state.logged_in:
    upload_and_chatbot_page()
else:
    login_page()
