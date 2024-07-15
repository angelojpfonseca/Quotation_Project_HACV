# data_ingestion.py

import os
import logging
from typing import List, Dict
from pypdf import PdfReader
from langchain.docstore.document import Document

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_pdf(file_path: str) -> str:
    """
    Load a PDF file and extract its text content.
    """
    try:
        logger.info(f"Loading PDF: {file_path}")
        with open(file_path, 'rb') as file:
            pdf = PdfReader(file)
            text = ""
            for page_num, page in enumerate(pdf.pages, 1):
                logger.debug(f"Processing page {page_num} of {file_path}")
                text += page.extract_text() + "\n"  # Add a newline between pages
        logger.info(f"Successfully extracted text from {file_path}")
        return text.strip()  # Remove leading/trailing whitespace
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {str(e)}")
        raise

def process_pdf(file_path: str, manufacturer: str) -> Document:
    """
    Process a single PDF file and return a Document object.
    """
    try:
        text = load_pdf(file_path)
        return Document(
            page_content=text,
            metadata={
                "source": file_path,
                "filename": os.path.basename(file_path),
                "manufacturer": manufacturer
            }
        )
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return None

def ingest_data() -> Dict[str, List[Document]]:
    """
    Ingest data from specified PDF files.
    """
    all_documents = {}
    
    # Define the paths and manufacturers
    pdf_paths = {
        "Daikin": r"B:\Angelo Data\GitHub\Quotation_Project_HACV\data\daikin\Split.pdf",
        "Melco": r"B:\Angelo Data\GitHub\Quotation_Project_HACV\data\melco\Mr. Slim.pdf"
    }

    for manufacturer, path in pdf_paths.items():
        logger.info(f"Processing {manufacturer} PDF: {path}")
        document = process_pdf(path, manufacturer)
        if document:
            all_documents[manufacturer] = [document]
            logger.info(f"Successfully processed {manufacturer} PDF")
        else:
            logger.warning(f"Failed to process {manufacturer} PDF")

    logger.info(f"Data ingestion complete. Processed {len(all_documents)} manufacturers.")
    return all_documents

if __name__ == "__main__":
    try:
        ingested_data = ingest_data()
        for manufacturer, docs in ingested_data.items():
            print(f"Manufacturer: {manufacturer}, Documents: {len(docs)}")
            # Print a sample of the extracted text
            if docs:
                print(f"Sample text from {manufacturer}:")
                print(docs[0].page_content[:500] + "...")  # Print first 500 characters
    except Exception as e:
        logger.error(f"An error occurred during data ingestion: {str(e)}")