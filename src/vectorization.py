import os
import logging
from typing import List, Dict, Any
import voyageai
from dotenv import load_dotenv
from document_processing import process_manufacturer_data
from data_ingestion import ingest_data
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Voyage AI client
voyage_client = voyageai.Client(api_key=os.getenv('VOYAGE_API_KEY'))

def vectorize_chunks(chunks: List[Document], batch_size: int = 128) -> List[Dict[str, Any]]:
    """
    Convert text chunks into vector embeddings using Voyage AI.

    Args:
        chunks (List[Document]): A list of Document objects.
        batch_size (int): The number of chunks to process in each batch. Default is 128.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing the original content,
                              metadata, and the vector embedding.
    """
    logger.info(f"Vectorizing {len(chunks)} chunks using Voyage AI")
    vectorized_docs = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        
        # Prepare texts for batch embedding
        texts = [chunk.page_content for chunk in batch]

        try:
            # Get embeddings from Voyage AI in batch
            result = voyage_client.embed(texts, model="voyage-2", input_type="document")
            
            for chunk, embedding in zip(batch, result.embeddings):
                vectorized_docs.append({
                    "content": chunk.page_content,
                    "metadata": chunk.metadata,
                    "vector": embedding
                })
            logger.info(f"Successfully vectorized batch of {len(batch)} chunks")
        except Exception as e:
            logger.error(f"Error vectorizing batch: {str(e)}")

    logger.info(f"Finished vectorizing. Total vectorized documents: {len(vectorized_docs)}")
    return vectorized_docs

def process_and_vectorize_data(processed_data: Dict[str, List[Document]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process and vectorize data for all manufacturers.

    Args:
        processed_data (Dict[str, List[Document]]): A dictionary where keys are manufacturer names
                                                    and values are lists of Document objects.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary with the same structure as the input, but with
                                         documents converted to vectorized format.
    """
    vectorized_data = {}
    for manufacturer, chunks in processed_data.items():
        logger.info(f"Vectorizing documents for manufacturer: {manufacturer}")
        try:
            vectorized_docs = vectorize_chunks(chunks)
            vectorized_data[manufacturer] = vectorized_docs
            logger.info(f"Vectorized {len(vectorized_docs)} documents for {manufacturer}")
        except Exception as e:
            logger.error(f"Error vectorizing documents for {manufacturer}: {str(e)}")

    return vectorized_data

if __name__ == "__main__":
    try:
        # Ingest the data
        ingested_data = ingest_data()
        
        # Process the ingested data
        processed_data = process_manufacturer_data(ingested_data)
        
        # Vectorize the processed data
        vectorized_data = process_and_vectorize_data(processed_data)

        # Print some information about the vectorized data
        for manufacturer, docs in vectorized_data.items():
            print(f"Manufacturer: {manufacturer}, Vectorized Documents: {len(docs)}")
            if docs:
                print(f"Sample vector length: {len(docs[0]['vector'])}")
                print(f"Sample metadata: {docs[0]['metadata']}")
                print("---")
    except Exception as e:
        logger.error(f"An error occurred during the vectorization process: {str(e)}")