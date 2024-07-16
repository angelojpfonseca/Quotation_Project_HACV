# data_pipeline_examiner.py

import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from mongodb_integration import MongoDBHandler
from dotenv import load_dotenv

load_dotenv()

def read_pdf(file_path):
    print(f"Reading PDF: {file_path}")
    with open(file_path, 'rb') as file:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def process_text(text, chunk_size=1000, chunk_overlap=200):
    print("Processing text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

def store_in_mongodb(mongo_handler, manufacturer, chunks):
    print(f"Storing {len(chunks)} chunks for {manufacturer} in MongoDB...")
    documents = [
        {
            "content": chunk,
            "metadata": {"manufacturer": manufacturer}
        } for chunk in chunks
    ]
    mongo_handler.store_vectorized_data({manufacturer: documents})

def retrieve_from_mongodb(mongo_handler, manufacturer):
    print(f"Retrieving documents for {manufacturer} from MongoDB...")
    return mongo_handler.get_all_documents(manufacturer)

def main():
    base_path = "B:\\Angelo Data\\GitHub\\Quotation_Project_HACV\\data"
    manufacturers = ["daikin", "melco"]
    
    mongo_handler = MongoDBHandler()
    mongo_handler.connect()

    for manufacturer in manufacturers:
        manufacturer_path = os.path.join(base_path, manufacturer)
        for filename in os.listdir(manufacturer_path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(manufacturer_path, filename)
                
                # Step 1: Read PDF
                raw_text = read_pdf(file_path)
                print(f"Raw text sample:\n{raw_text[:500]}...\n")

                # Step 2: Process Text
                chunks = process_text(raw_text)
                print(f"Processed {len(chunks)} chunks. Sample chunk:\n{chunks[0]}\n")

                # Step 3: Store in MongoDB
                store_in_mongodb(mongo_handler, manufacturer, chunks)

                # Step 4: Retrieve from MongoDB
                retrieved_docs = retrieve_from_mongodb(mongo_handler, manufacturer)
                print(f"Retrieved {len(retrieved_docs)} documents from MongoDB.")
                if retrieved_docs:
                    print(f"Sample retrieved document:\n{retrieved_docs[0]}\n")

    mongo_handler.close_connection()

if __name__ == "__main__":
    main()