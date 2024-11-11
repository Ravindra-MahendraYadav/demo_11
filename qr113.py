import streamlit as st
from langchain_community.embeddings.openai import OpenAIEmbeddings  # Use langchain_community version
from langchain_community.vectorstores import Chroma  # Updated import path
import os
import qrcode
from io import BytesIO
from PIL import Image
import speech_recognition as sr
from dotenv import load_dotenv

# Function to load document based on file type
def load_document(file):
    name, extension = os.path.splitext(file)
    if extension == ".pdf":
        from langchain.document_loaders import PyPDFLoader
        print(f'loading {file}')
        loader = PyPDFLoader(file)
    elif extension == ".docx":
        from langchain.document_loaders import Docx2txtLoader
        print(f'loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == ".txt":
        from langchain.document_loaders import TextLoader
        print(f'loading {file}')
        loader = TextLoader(file)
    else:
        print("Document Type Not supported")
        return None
    data = loader.load()
    return data

def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    chroma_db_path = os.path.join(os.path.expanduser("~"), "chroma_db_storage")  # Ensure persistence
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=chroma_db_path)
    return vector_store

def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    if vector_store is None:
        raise ValueError("Vector store is not initialized. Please upload and process a document first.")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})

    if retriever is None:
        raise ValueError("Failed to create retriever. Check if vector store has data.")
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = chain.run(q)
    return answer

# Function to create and display QR code
def generate_qr_code(url, size=(250, 250)):
    qr = qrcode.make(url)
    qr = qr.resize(size, Image.Resampling.LANCZOS)  # Resize the QR code
    buf = BytesIO()
    qr.save(buf, format="PNG")
    buf.seek(0)
    return buf

# Function to capture speech input and return text
def capture_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak your question")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            st.success("Speech recognized successfully!")
            return text
        except sr.UnknownValueError:
            st.error("Could not understand the audio")
        except sr.RequestError:
            st.error("Error in processing the request")
        except sr.WaitTimeoutError:
            st.warning("No speech detected within the timeout period")
    return None

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv("Ravindra.env")
    
    st.image("Meril.jpg")
    st.subheader("Mizzo SmartGuide")

    # Sidebar for API key, file upload, and additional settings
    with st.sidebar:
        # Input for API key
        api_key = st.text_input("OPENAI API KEY:", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key  # Fixed typo here

        # File upload option
        uploaded_file = st.file_uploader("Upload a File:", type=["pdf", "docx", "txt"])
        
        # Number input for chunk size and k
        chunk_size = st.number_input("chunk size:", min_value=100, max_value=2048, value=512)
        k = st.number_input("k", min_value=1, max_value=20, value=3)

        # Button to trigger data addition
        add_data = st.button("Add Data")
        
        # File processing logic
        if uploaded_file and add_data:
            with st.spinner("Reading, Chunking, and Embedding file..."):
                # Read and save file
                bytes_data = uploaded_file.read()
                file_name = os.path.join("./", uploaded_file.name)
                with open(file_name, "wb") as f:
                    f.write(bytes_data)

                # Load, chunk, and embed data
                data = load_document(file_name)
                if data:
                    chunks = chunk_data(data, chunk_size=chunk_size)
                    st.write(f'chunk size: {chunk_size}, chunks: {len(chunks)}')
                    vector_store = create_embeddings(chunks)

                    # Store vector data in session state
                    st.session_state.vs = vector_store
                    st.success("File Uploaded, Chunked, Embedded Successfully")

    # QR code dropdown button
    with st.expander("📱 Run On Your Mobile: "):
        url = "http://smartmanual.streamlit.app"  # Replace with your actual deployment URL
        qr_image = generate_qr_code(url, size=(250, 250))
        st.image(qr_image, caption="Scan to access deployment", use_column_width=False, width=250)

    # Dropdown for asking questions
    with st.expander("💬 Ask Your Question"):
        question = st.text_input("Type your question here:")
        if st.button("🎤 Capture Speech"):
            st.warning("This feature is not included yet")

    if question:
        if "vs" in st.session_state:
            vector_store = st.session_state["vs"]
            st.write(f'k: {k}')
            answer = ask_and_get_answer(vector_store, question, k)
            st.text_area("LLM Answer:", value=answer , height=500)
