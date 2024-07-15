import logging
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from data_ingestion import ingest_data  # Import the ingest_data function

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_documents(documents: List[Document], 
                    chunk_size: int = 1000, 
                    chunk_overlap: int = 200) -> List[Document]:
    """
    Split the input documents into smaller chunks.

    Args:
        documents (List[Document]): List of Document objects to be split.
        chunk_size (int): The size of each chunk in characters. Default is 1000.
        chunk_overlap (int): The overlap between chunks in characters. Default is 200.

    Returns:
        List[Document]: A list of Document objects representing the split chunks.
    """
    logger.info(f"Splitting {len(documents)} documents into chunks (size: {chunk_size}, overlap: {chunk_overlap})")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    split_docs = []
    for doc in documents:
        try:
            chunks = text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                split_docs.append(Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                ))
        except Exception as e:
            logger.error(f"Error splitting document {doc.metadata.get('source', 'Unknown')}: {str(e)}")

    logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    return split_docs

def process_manufacturer_data(manufacturer_data: Dict[str, List[Document]]) -> Dict[str, List[Document]]:
    """
    Process the data for all manufacturers by splitting their documents.

    Args:
        manufacturer_data (Dict[str, List[Document]]): A dictionary where keys are manufacturer names
                                                       and values are lists of Document objects.

    Returns:
        Dict[str, List[Document]]: A dictionary with the same structure as the input, but with
                                   documents split into chunks.
    """
    processed_data = {}
    for manufacturer, docs in manufacturer_data.items():
        logger.info(f"Processing documents for manufacturer: {manufacturer}")
        try:
            processed_docs = split_documents(docs)
            processed_data[manufacturer] = processed_docs
            logger.info(f"Processed {len(docs)} documents into {len(processed_docs)} chunks for {manufacturer}")
        except Exception as e:
            logger.error(f"Error processing documents for {manufacturer}: {str(e)}")

    return processed_data

if __name__ == "__main__":
    try:
        # Use the actual data from data_ingestion.py
        ingested_data = ingest_data()
        processed_data = process_manufacturer_data(ingested_data)
        
        for manufacturer, docs in processed_data.items():
            print(f"Manufacturer: {manufacturer}, Processed Chunks: {len(docs)}")
            if docs:
                print(f"Sample chunk from {manufacturer}:")
                print(docs[0].page_content[:200] + "...")  # Print first 200 characters of the first chunk
                print(f"Metadata of the first chunk: {docs[0].metadata}")
                print("---")
    except Exception as e:
        logger.error(f"An error occurred during document processing: {str(e)}")