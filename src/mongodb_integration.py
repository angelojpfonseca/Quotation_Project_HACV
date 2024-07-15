import os
import logging
from typing import Dict, List, Any
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, OperationFailure
from bson.objectid import ObjectId
from dotenv import load_dotenv

print(f"OS environment MONGODB_URI: {os.environ.get('MONGODB_URI')}")
# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB connection string should be stored in an environment variable
MONGODB_URI = os.getenv('MONGODB_URI')
if MONGODB_URI:
    masked_uri = MONGODB_URI.split('@')
    if len(masked_uri) > 1:
        masked_uri = f"{'@'.join(masked_uri[:-1])}@{'*' * len(masked_uri[-1])}"
    else:
        masked_uri = '*' * len(MONGODB_URI)
    print(f"Loaded MongoDB URI: {masked_uri}")
else:
    print("MongoDB URI not found in .env file")
DB_NAME = 'product_comparison'  # You can change the database name

class MongoDBHandler:
    def __init__(self):
        self.client = None
        self.db = None

    def connect(self):
        """Establish a connection to MongoDB."""
        try:
            self.client = MongoClient(MONGODB_URI)
            self.db = self.client[DB_NAME]
            # The ismaster command is cheap and does not require auth.
            self.client.admin.command('ismaster')
            logger.info("Successfully connected to MongoDB")
        except ConnectionFailure:
            logger.error("Server not available")
            raise
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {str(e)}")
            raise

    def close_connection(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("Closed MongoDB connection")

    def create_vector_index(self, collection_name: str):
        """Create a vector index for similarity search."""
        try:
            self.db[collection_name].create_index([("vector", ASCENDING)], name="vector_index")
            logger.info(f"Created vector index for collection {collection_name}")
        except OperationFailure as e:
            logger.error(f"Error creating vector index: {str(e)}")

    def store_vectorized_data(self, vectorized_data: Dict[str, List[Dict[str, Any]]]):
        """Store the vectorized data in MongoDB."""
        for manufacturer, docs in vectorized_data.items():
            collection = self.db[f"{manufacturer}_products"]
            try:
                result = collection.insert_many(docs)
                logger.info(f"Inserted {len(result.inserted_ids)} documents for {manufacturer}")
                self.create_vector_index(f"{manufacturer}_products")
            except Exception as e:
                logger.error(f"Error inserting documents for {manufacturer}: {str(e)}")

    def retrieve_similar_documents(self, manufacturer: str, query_vector: List[float], limit: int = 5):
        """Retrieve similar documents based on vector similarity."""
        collection = self.db[f"{manufacturer}_products"]
        try:
            similar_docs = collection.aggregate([
                {
                    "$search": {
                        "index": "vector_index",
                        "knnBeta": {
                            "vector": query_vector,
                            "path": "vector",
                            "k": limit
                        }
                    }
                },
                {
                    "$project": {
                        "content": 1,
                        "metadata": 1,
                        "score": {"$meta": "searchScore"}
                    }
                }
            ])
            return list(similar_docs)
        except Exception as e:
            logger.error(f"Error retrieving similar documents: {str(e)}")
            return []

    def update_document(self, manufacturer: str, document_id: str, update_data: Dict[str, Any]):
        """Update a specific document."""
        collection = self.db[f"{manufacturer}_products"]
        try:
            result = collection.update_one({"_id": ObjectId(document_id)}, {"$set": update_data})
            if result.modified_count > 0:
                logger.info(f"Updated document {document_id} for {manufacturer}")
            else:
                logger.warning(f"No document found with id {document_id} for {manufacturer}")
        except Exception as e:
            logger.error(f"Error updating document {document_id} for {manufacturer}: {str(e)}")

    def delete_document(self, manufacturer: str, document_id: str):
        """Delete a specific document."""
        collection = self.db[f"{manufacturer}_products"]
        try:
            result = collection.delete_one({"_id": ObjectId(document_id)})
            if result.deleted_count > 0:
                logger.info(f"Deleted document {document_id} for {manufacturer}")
            else:
                logger.warning(f"No document found with id {document_id} for {manufacturer}")
        except Exception as e:
            logger.error(f"Error deleting document {document_id} for {manufacturer}: {str(e)}")

    def get_all_documents(self, manufacturer: str, limit: int = 100):
        """Retrieve all documents for a manufacturer, with a limit."""
        collection = self.db[f"{manufacturer}_products"]
        try:
            documents = collection.find().limit(limit)
            return list(documents)
        except Exception as e:
            logger.error(f"Error retrieving documents for {manufacturer}: {str(e)}")
            return []

def main():
    mongo_handler = MongoDBHandler()
    
    try:
        mongo_handler.connect()

        # Example usage (uncomment and modify as needed):
        # from vectorization import process_and_vectorize_data
        # from document_processing import process_manufacturer_data
        # from data_ingestion import ingest_data

        # # Ingest and process data
        # ingested_data = ingest_data()
        # processed_data = process_manufacturer_data(ingested_data)
        # vectorized_data = process_and_vectorize_data(processed_data)

        # # Store data in MongoDB
        # mongo_handler.store_vectorized_data(vectorized_data)

        # # Test retrieval
        # sample_vector = vectorized_data['Daikin'][0]['vector']
        # similar_docs = mongo_handler.retrieve_similar_documents('Daikin', sample_vector)
        # print(f"Retrieved {len(similar_docs)} similar documents")

        # # Test update
        # if similar_docs:
        #     doc_id = str(similar_docs[0]['_id'])
        #     mongo_handler.update_document('Daikin', doc_id, {"updated_field": "new_value"})

        # # Test deletion
        # if similar_docs:
        #     doc_id = str(similar_docs[0]['_id'])
        #     mongo_handler.delete_document('Daikin', doc_id)

        # # Test get all documents
        # all_docs = mongo_handler.get_all_documents('Daikin', limit=10)
        # print(f"Retrieved {len(all_docs)} documents for Daikin")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        mongo_handler.close_connection()

if __name__ == "__main__":
    main()