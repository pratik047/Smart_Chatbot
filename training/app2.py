import logging
import spacy
import streamlit as st
from dotenv import load_dotenv
import networkx as nx
from pyvis.network import Network
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG_Application")

# Load environment variables
load_dotenv()

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")
logger.info("SpaCy model loaded successfully.")

# Streamlit application title
st.title("RAG Application with Knowledge Graph")

# Load PDF and process documents
file_path = "MC_ans.pdf"
try:
    loader = PyPDFLoader(file_path)
    data = loader.load()
    logger.info(f"Loaded {len(data)} pages from PDF.")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)
    logger.info(f"Split documents into {len(docs)} chunks.")

    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 10}
    )

    # Load LLM (Gemini Model)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None
    )

    # Build Knowledge Graph
    def build_knowledge_graph(docs):
        graph = nx.DiGraph()

        for doc in docs:
            doc_nlp = nlp(doc.page_content)
            for ent in doc_nlp.ents:
                graph.add_node(ent.text, label=ent.label_)

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
                            )
        return graph

    # Render Knowledge Graph
    def render_knowledge_graph(graph):
        net = Network(height="500px", width="100%", directed=True)

        for node, data in graph.nodes(data=True):
            net.add_node(node, title=f"{node} ({data.get('label', 'Unknown')})")

        for src, dest, data in graph.edges(data=True):
            relation = data.get("relation", "")
            net.add_edge(src, dest, title=relation)

        return net.generate_html("knowledge_graph.html")

    # Build and display the knowledge graph
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
    logger.error(f"Error during RAG process: {e}")
    st.error(f"An error occurred: {e}")
