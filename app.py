import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from streamlit import session_state as ss
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import pyttsx3
import fitz
from PIL import Image
import io



# Load environment variables

# Set API keys
grouq_key = st.secrets["gr_key"]
os.environ["GROQ_API_KEY"] = grouq_key

# Initialize LLM
llm = ChatGroq( key =grouq_key, model="Gemma2-9b-It")



# Streamlit app setup
st.set_page_config(page_title="Study Helper", page_icon="📚")

st.title("📚 Let's Make Study Easy!")

# Initialize session state
if 'pdf_ref' not in ss:
    ss.pdf_ref = None
if 'chat_history' not in ss:
    ss.chat_history = []
if "store" not in ss:
    ss.store = {}
if "documents" not in ss:
    ss.documents = None  # To store processed documents
if "response" not in ss:
    ss.response = None  # To store generated response
if "play" not in ss:
    ss.play = False  # To track play state




# Sidebar for file uploads and session management
with st.sidebar:
    st.title("📂 Upload Your Files")
    session_id = st.text_input("Session ID", value="default", help="Enter a unique session ID to track your progress.")
    uploaded_files = st.file_uploader("Upload PDFs", type=['pdf'], accept_multiple_files=True)
    
    st.markdown("---")
    st.markdown("#### Instructions")
    st.markdown("""
    1. Upload your PDF files.
    2. Enter a session ID (optional).
    3. Ask your questions in the main section.
    """)

# Only process PDFs once and store in session state
if uploaded_files and ss.documents is None:
    documents = []
  
    
    st.markdown("### Uploaded Files")
    with st.spinner('Processing uploaded files...'):
        for file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as f:
                f.write(file.getvalue())
            file_name = file.name

            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

           
          
            os.remove(temppdf)

            st.success(f"Processed {file_name} successfully!")

    # Store processed documents in session state to avoid reprocessing
    ss.documents = documents


# Only if PDFs are processed, proceed to the main logic
if ss.documents:
    contextualize_q_system_prompt = (
        "Given a chat history and latest user question"
        "which might reference context in the chat history."
        "Formulate the standalone question which can be understood"
        "without the chat history. Do not answer the question"
        "just reformulate it if needed; otherwise, return it as it is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    qa_system_prompt = (
        "You are the assistant for question answering task."
        "Use the following retrieved context to answer"
        "the question. If you don't know the answer, say that you don't know."
        "Keep the answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Splitting documents for vector storage
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splitted_docs = splitter.split_documents(ss.documents)
    embedding = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(splitted_docs, embedding=embedding)
    retriever = vectordb.as_retriever()

    # Create retrieval chain
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    stuff_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, stuff_docs_chain)

    # Session history management
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in ss.store:
            ss.store[session_id] = ChatMessageHistory()
        return ss.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history=get_session_history,
        input_messages_key="input",
        output_messages_key="answer",
        history_messages_key="chat_history"
    )

    # User input for questions
    user_input = st.text_input("💬 Ask your question", placeholder="Type your question here...", key="user_input")

    # Response section
    if user_input:
        with st.spinner('Thinking...'):
            try:
               
                if not ss.response or 'user_input' in st.session_state:
                    ss.response = conversational_rag_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": session_id}}
                    )
                st.write(f"**Answer:** {ss.response['answer']}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Display play button after generating a response
    if ss.response:
        if st.button("▶️"):
            ss.play = True  # Set play state to true when button is pressed

        # Play the response using pyttsx3 only if play button is pressed
        if ss.play:
            engine = pyttsx3.init()
            engine.say(ss.response['answer'])
            engine.runAndWait()
            ss.play = False  # Reset play state after playing

    
    
else:
    st.warning("Please upload PDF files to start the process.")
