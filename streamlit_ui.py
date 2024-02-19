############# Streamlit UI For Testing Purpose #############

#Importing the required libraries

import streamlit as st
import chromadb
from PyPDF2 import PdfReader 
from chunking import gpt_chunking 
from chunking import qa_processing 
import os
from os. path import abspath
import re
# from 1lmsherpa.readers import LayoutPDFReader
# from 11msherpa.readers import layout_reader
from sentence_transformers import SentenceTransformer, util
db_path = "/home/roy_aakash/Downloads/mirror2_local/mirror2/docs_vectorDB"
client = chromadb.PersistentClient(path=r"/home/roy_aakash/Downloads/mirror2_local/mirror2/docs_vectorDB")
embedder = SentenceTransformer('all-MiniLM-L6-v2')


#Function to upload and process the PDF

def upload_and_process_pdf():
    # Write code to upload and process the PDF
    uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")

    options = ["AI Chunking","Slicing Chunking", "LLAMAsherpa", "Recursive chunking"]
    selected_strategy = st.selectbox("Select Chunking Strategy", options)
    # Save processed data in the database
    if selected_strategy=="Slicing Chunking":

        if uploaded_file is not None:

            with st.form(key='my_form'):
                db_name = st.text_input("Enter the database name")
                
                submit_button = st.form_submit_button(label='Submit')

            if submit_button:

                if not os.path.isdir(db_path):
                    st.error('The provided folder does not exist. Please provide a valid folder path.')

                reader = PdfReader(uploaded_file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()

                #st.write(uploaded_file, 'File loaded successfully!')
                st.markdown('Processing started...')
                qa_processing.chunk_and_db_1(text, db_name, db_path)
                #parent_chunk, child_chunk = gpt_chunking.process(text=text)
                st.markdown('Saved into database...')




# Function to display available databases

def show_available_databases_1():
    # Write code to fetch and display available databases
    databases = []
    for item in client.list_collections():
        databases.append(str(list(item)[0][1]))
    selected_database = st.sidebar.selectbox("Select Database", databases)
    collection_chunk = client.get_collection(str(selected_database))

    return collection_chunk


# Function to display chat UI
def display_chat_ui_1(collection_chunk):
    # Write code to display chat UI based on data source
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response, top_chunks = qa_processing.get_answer_1(prompt, collection_chunk, top_k=5)
            message_placeholder.markdown(response)
            #message_placeholder.markdown("Top chunks:")
            #message_placeholder.markdown(top_chunks)

        st.session_state.messages.append({"role": "assistant", "content": response})

    if st.button('Clear Chat'):
        st.session_state.messages = [] 

# Main function to create the Streamlit UI
def main():
    st.title("Welcome to Carelon Document Inquiry App")

    page_options = ["Upload a PDF", "Available Databases"]
    page_selection = st.sidebar.radio("Select Page", page_options)

    if page_selection == "Upload a PDF":
        upload_and_process_pdf()
        #display_chat_ui(collection_child, collection_parent)
    else:
        collection_chunk = show_available_databases_1()
        display_chat_ui_1(collection_chunk)

if __name__ == "__main__":
    main()