import os
import io
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
import anthropic
from dotenv import load_dotenv
import base64

# Load environment variables
load_dotenv()

# MongoDB setup
MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI)
db = client.pdf_database
pdf_collection = db.pdf_chunks

# Anthropic setup
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'excluded_pdfs' not in st.session_state:
    st.session_state.excluded_pdfs = []
if 'query' not in st.session_state:
    st.session_state.query = ""

def process_pdf(file, selected_ranges):
    try:
        pdf = PdfReader(file)
        
        text = ""
        for name, start_page, end_page in selected_ranges:
            for page_num in range(start_page - 1, min(end_page, len(pdf.pages))):
                text += pdf.pages[page_num].extract_text() + "\n"
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = text_splitter.split_text(text)
        
        # Store chunks in MongoDB
        pdf_collection.delete_many({"filename": file.name})  # Remove existing chunks for this file
        for chunk in chunks:
            pdf_collection.insert_one({
                "content": chunk, 
                "filename": file.name,
                "sections": [name for name, _, _ in selected_ranges]
            })
        
        return f"Processed and stored {len(chunks)} chunks from {file.name}"
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def query_pdf_content(query, excluded_pdfs, max_tokens=8000):
    all_chunks = list(pdf_collection.find({"filename": {"$nin": excluded_pdfs}}, {"content": 1, "_id": 0}))
    
    context = ""
    for chunk in all_chunks:
        if len(context) + len(chunk['content']) > max_tokens:
            break
        context += chunk['content'] + "\n"
    
    chat_history = "\n".join([f"User: {msg['user']}\nAssistant: {msg['assistant']}" for msg in st.session_state.chat_history])
    
    system_prompt = f"""You are an AI assistant that answers questions based on the following PDF content and chat history:

PDF Content:
{context}

Chat History:
{chat_history}

Answer the user's questions based on this information. If asked to create a table, use markdown format to generate it. Be comprehensive and detailed in your responses."""

    try:
        response = anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": query
                }
            ]
        )
        return response.content[0].text
    except anthropic.BadRequestError as e:
        return f"An error occurred: {str(e)}"

def display_pdf(file):
    try:
        file.seek(0)  # Reset file pointer to the beginning
        base64_pdf = base64.b64encode(file.read()).decode('utf-8')
        pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf">'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")

# Streamlit UI
st.set_page_config(layout="wide")

col1, col2 = st.columns([2, 1])

with col1:
    st.title("PDF Chatbot System")

    # File upload and processing
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        st.write("PDF uploaded successfully")
        try:
            pdf = PdfReader(uploaded_file)
            st.write(f"Total pages: {len(pdf.pages)}")

            st.subheader("Select Page Ranges to Process")
            num_ranges = st.number_input("Number of ranges to process", min_value=1, value=1)
            selected_ranges = []

            for i in range(num_ranges):
                col_name, col_start, col_end = st.columns(3)
                with col_name:
                    name = st.text_input(f"Name for range {i+1}", key=f"name_{i}")
                with col_start:
                    start_page = st.number_input(f"Start page", min_value=1, max_value=len(pdf.pages), key=f"start_{i}")
                with col_end:
                    end_page = st.number_input(f"End page", min_value=start_page, max_value=len(pdf.pages), key=f"end_{i}")
                selected_ranges.append((name, start_page, end_page))

            if st.button("Process Selected Ranges"):
                result = process_pdf(uploaded_file, selected_ranges)
                st.success(result)
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")

    # Chat interface
    st.subheader("Chat")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            st.write(f"**User:** {message['user']}")
            st.markdown(f"**Assistant:** {message['assistant']}")
            st.write("---")

    # Manage the query input and sending
    query = st.text_input("Ask a question about the PDFs:", key="query_input")
    if st.button("Send"):
        st.session_state.query = query
        answer = query_pdf_content(st.session_state.query, st.session_state.excluded_pdfs)
        st.session_state.chat_history.append({"user": st.session_state.query, "assistant": answer})
        st.session_state.query = ""  # Clear the query from session state
        st.experimental_rerun()

# JavaScript to capture Enter key press and trigger Send button
st.markdown("""
<script>
document.getElementById('query_input').addEventListener('keydown', function(event) {
    if (event.key === 'Enter') {
        document.querySelector('button[data-testid="stButton"]').click();
    }
});
</script>
""", unsafe_allow_html=True)

with col2:
    if uploaded_file is not None:
        st.subheader("PDF Viewer")
        try:
            display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"Error displaying PDF: {str(e)}")

    # Stored PDFs and Sections
    st.subheader("Stored PDFs and Sections")
    stored_pdfs = pdf_collection.distinct("filename")
    for pdf in stored_pdfs:
        if st.checkbox(f"Include {pdf}", key=f"include_{pdf}"):
            if pdf not in st.session_state.excluded_pdfs:
                st.session_state.excluded_pdfs.append(pdf)
        else:
            if pdf in st.session_state.excluded_pdfs:
                st.session_state.excluded_pdfs.remove(pdf)

        st.write(f"PDF: {pdf}")
        sections = pdf_collection.distinct("sections", {"filename": pdf})
        for section in sections:
            st.write(f"- {section}")
        if st.button(f"Delete {pdf}", key=f"delete_{pdf}"):
            pdf_collection.delete_many({"filename": pdf})
            st.success(f"Deleted {pdf}")
            st.experimental_rerun()

# Connection status
st.sidebar.title("Connection Status")
try:
    client.admin.command('ismaster')
    st.sidebar.success("Connected to MongoDB")
except Exception as e:
    st.sidebar.error(f"Failed to connect to MongoDB: {e}")

try:
    anthropic_client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1,
        system="Test",
        messages=[{"role": "user", "content": "Test"}]
    )
    st.sidebar.success("Anthropic API is working")
except Exception as e:
    st.sidebar.error(f"Failed to connect to Anthropic API: {e}")
