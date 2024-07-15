import streamlit as st
from src.data_ingestion import process_manufacturer_pdfs
from src.document_processing import split_texts
from src.vectorization import vectorize_chunks
from src.database import insert_documents, query_documents
from src.llm_analysis import analyze_product, compare_products

st.title("Product Comparison App")

# File upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Process the uploaded file
    text = process_manufacturer_pdfs([uploaded_file])[0]
    chunks = split_texts([text])
    vectors = vectorize_chunks(chunks)
    
    # Store in database
    insert_documents('uploaded_products', [{'text': chunk, 'vector': vector} for chunk, vector in zip(chunks, vectors)])
    
    # Analyze the product
    analysis = analyze_product(text)
    st.write("Product Analysis:", analysis)
    
    # Compare with other products
    manufacturer = st.selectbox("Select a manufacturer to compare with", ["Daikin", "Melco"])
    comparison_product = query_documents(f'{manufacturer.lower()}_products', {}).limit(1)[0]
    comparison = compare_products(text, comparison_product['text'])
    st.write("Product Comparison:", comparison)

# Run the Streamlit app:
# streamlit run app.py