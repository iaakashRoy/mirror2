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
client = chromadb.Persistentclient(path=r"/home/roy_aakash/Downloads/mirror2_local/mirror2/docs_vectorDB")
embedder = SentenceTransformer ('all-MiniLM-L6-v2')



#Function to upload and process the PDF

def upload_and_process_pdf():

    uploaded_file = st. file_uploader ('Choose your pdf file', type="pdf")
    options = ["AI Chunking", "LLAMAsherpa", "Recursive chunking"]
    selected_strategy = st. selectbox("Select Chunking Strategy", options)
    # Save processed data in the database
    if selected_strategy=="AI Chunking":
        #collection_child, collection parent = ll, l]
        if uploaded_file is not None:
            with st. form(key='my_form'):
                db_name = st.text_input("Enter the database name")
                submit_button = st. form_submit_button(label='Submit')
            if submit_button:
                child_name = db_name+'_C'
                parent_name = db_name+'_P'
                if not os.path. isdir(db_path):
                    st.error('The provided folder does not exist. Please provide a valid folder path')
            reader = PdfReader (uploaded_file)
            text = ''
            for page in reader. pages:
                text += page. extract_text()
            #st. write(uploaded_file, 'File loaded successfully!')
            st.markdown ('Processing started...')
            qa_processing.chunk_and_db(text, child_name, parent_name, db_path)
            #parent_chunk, child_chunk = gpt_chunking-process(text=text)
            st.markdown('Saved into database...')



# Function to display available databases
            
def show_available_databases():
    
    databases = []
    for item in client.list_collections():
        if str(list(item)[0][1][: -2]) not in databases: 
            databases. append (str(list(item)[0][1][: -2]))
    selected_database = st.sidebar.selectbox("Select Database", databases)
    collection_child = client.get_collection (str (selected_database) + '_C')
    collection_parent = client.get_collection (str(selected_database) + '_P')
    return collection_child, collection_parent

# Function to display chat UI
def display_chat_ui(collection_child, collection_parent):

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st. session_state.messages:
        with st.chat_message (message["role"]):
            st.markdown(message ["content"] )
    if prompt := st.chat_input ("what is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st. chat_message("user"):
            st .markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st. empty()
            response, top_childs, updated_top_parents = qa_processing.get_answer(prompt, collection_child, collection_parent, child_top_k=5, parent_top_k=4)
            message_placeholder.markdown(response)
        try:
            for context in updated_top_parents [:3]:
                st.markdown(qa_processing.convert_dict_to_text(eval (context)))
        except:
            st.session_state.messages.append({"role": "assistant", "content": response})
    if st.button('Clear Chat'):
        st.session_state.messages = []
# Main function to create the Streamlit
                              
                              # Main function to create the Streamlit UI
def main():
    st. title("welcome to Carelon Document Inquiry App")

    page_options = ["Upload a PDF", "Available Databases"]
    page_selection = st. sidebar.radio("Select Page", page_options)

    if page_selection == "Upload a PDF":
        upload_and_process_pdf ()
        #display_chat_ui(collection child, collection parent) 
        
    else:
        collection_child, collection_parent = show_available_databases()
        display_chat_ui(collection_child, collection_parent)
if __name__ == "__main__":
    main ()