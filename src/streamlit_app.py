# streamlit_app.py

import streamlit as st
import anthropic
from mongodb_integration import MongoDBHandler
from dotenv import load_dotenv
import os
import pandas as pd

# Load environment variables
load_dotenv()

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Initialize MongoDB handler
mongo_handler = MongoDBHandler()
mongo_handler.connect()

st.title("Product Comparison Chatbot")

# Debug mode toggle
debug_mode = st.sidebar.checkbox("Debug Mode")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to query MongoDB and format results
def query_mongodb(manufacturer, limit=10):
    docs = mongo_handler.get_all_documents(manufacturer, limit)
    return [{"content": doc["content"], "metadata": doc["metadata"]} for doc in docs]

# Function to generate product comparison table
def generate_comparison_table(manufacturers):
    data = []
    for manufacturer in manufacturers:
        docs = query_mongodb(manufacturer, limit=5)
        for doc in docs:
            data.append({
                "Manufacturer": manufacturer,
                "Product": doc["metadata"].get("filename", "Unknown"),
                "Details": doc["content"][:100] + "..."  # Truncate content for brevity
            })
    return pd.DataFrame(data)

# Function to print sample data
def print_sample_data():
    manufacturers = ["Daikin", "Melco"]  # Add all your manufacturers here
    for manufacturer in manufacturers:
        docs = query_mongodb(manufacturer, limit=5)
        st.write(f"Sample data for {manufacturer}:")
        for doc in docs:
            st.write(f"Filename: {doc['metadata'].get('filename', 'Unknown')}")
            st.write(f"Content: {doc['content'][:200]}...")
            st.write("---")

# Button to print sample data
if st.button("Print Sample Data"):
    print_sample_data()

# Accept user input
if prompt := st.chat_input("What would you like to know about our products?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get chatbot response
    manufacturers = ["Daikin", "Melco"]  # Add all your manufacturers here
    context = f"You are a helpful assistant with knowledge about HVAC products from {', '.join(manufacturers)}. "
    context += "You have access to a MongoDB database with product information. "
    context += "Here's a sample of the data for each manufacturer:\n\n"
    
    for manufacturer in manufacturers:
        docs = query_mongodb(manufacturer, limit=10)
        context += f"{manufacturer} products:\n"
        for doc in docs:
            context += f"- Product: {doc['metadata'].get('filename', 'Unknown')}\n"
            context += f"  Details: {doc['content'][:500]}...\n\n"
    
    context += "\nWhen answering questions, use this product information. If you need more details or a comparison, say 'GENERATE_TABLE'."

    if debug_mode:
        st.sidebar.write("Context sent to Claude:")
        st.sidebar.text(context)

    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        system=context,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response_text = response.content[0].text
        st.markdown(response_text)
        
        # Check if the response includes the trigger to generate a table
        if "GENERATE_TABLE" in response_text:
            st.subheader("Product Comparison")
            comparison_table = generate_comparison_table(manufacturers)
            st.dataframe(comparison_table)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})

# Close MongoDB connection when the app is closed
mongo_handler.close_connection()