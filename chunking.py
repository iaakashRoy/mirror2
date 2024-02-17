############# Chunking Strategies #############


# Importing the required libraries

import openai 
import json
import os
import re
import pandas as pd
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
#from tqdm.notebook import tqdm
from stqdm import stqdm
from sentence_transformers import SentenceTransformer, util
import chromadb
import torch
#import get_llm
import local_llm
import json
import requests
import warnings
warnings. filterwarnings("ignore")
embedder = SentenceTransformer ('all-MiniLM-L6-v2')


class gpt_chunking:       


    #Function to slice the data according to the end of sentences given the range
    def slice_text(text, range_start, range_end):
        # Split the text into sentences
        sentences = text.split(". ")

        # Check if range indices are valid
        if range_start < 0 or range_end > len(sentences):
            return "Invalid range!"

        # Slice the sentences according to the given range
        sliced_text = ". ".join(sentences[range_start : range_end])

        # Append a dot at end if there was a dot at the end of last sentence
        if sentences[range_end - 1][-1] == ".":
            sliced_text += "."

        return sliced_text
    

class qa_processing:

    def make_categorize_conversation_prompt(query, top_search_results):
        instructions = f""" You're assigned a task based on the following query: {query}, and you're provided with a selection of key information from top search results: {top_search_results}. //
        As a chat agent working for a renowned health insurance company, it is expected that your response is concise, informative, and politely-worded. // 
        Your task is to formulate an accurate answer using data from the given search results, related specifically to the query. //
        Be sure to omit any irrelevant data from your composed answer, regardless of its presence in the top information list. //
        DO NOT introduce any external information not presented in the search results. This is crucial to prevent the dissemination of incorrect or misleading information. //
        IF NO INFORMATION IS FOUND IN THE GIVEN TOP_SEARCH_RESULTS LIST, MSIMPLY SAY: NO MATCHING DETAILS FOUND IN THE TEXT
        """
        return instructions

    def fetching_results_chid_n_parent(query, parent_corpus, child_corpus, parent_top_k=5, child_top_k=15, num_childs=4):
        child_corpus_embeddings = embedder.encode(child_corpus, convert_to_tensor=True)
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, child_corpus_embeddings)[0]
        top_results_c = torch.topk(cos_scores, k=child_top_k)
        
        top_parents = []
        
        top_res = []
        for i in top_results_c[1]:
            top_res.append(child_corpus[i])
            parent_idx = i//num_childs
            if parent_corpus[parent_idx] not in top_parents:
                top_parents.append(parent_corpus[parent_idx])
                
        top_parents_embeddings = embedder.encode(top_parents, convert_to_tensor=True)        
        cos_scores = util.cos_sim(query_embedding, top_parents_embeddings)[0]
        try:
            top_results_p = torch.topk(cos_scores, k=parent_top_k)
        except:
            top_results_p = torch.topk(cos_scores, k = len(top_parents_embeddings))
        
        for i in top_results_p[1]:
            top_res.append(top_parents[i])
        
        answer = gpt_chunking.get_completion(qa_processing.make_categorize_conversation_prompt(query, top_res))
        top_childs_ = []
        for score, idx in zip(top_results_c[0], top_results_c[1]):
            top_childs_.append((child_corpus[idx] , score))
        top_parents_ = []
        for score, idx in zip(top_results_p[0], top_results_p[1]):
            top_parents_.append((top_parents[idx], score))
        return answer, top_childs_ , top_parents_

    def child_id_to_parent_id(child_id, num_childs):
        c_id = int(child_id[2:])  #id67 => int 67 
        p_id = c_id//num_childs  # 67//4 = 16 : 0-3=0, 
        return 'id'+ str(p_id)  # id16

    def split_list(input_list, chunk_size):
        for i in range(0, len(input_list), chunk_size):
            yield input_list[i:i + chunk_size]

    def convert_dict_to_text(dictionary):
        text = ""
        for key, value in dictionary.items():
            if isinstance(value, list):
                text += f"\n{key}:\n"
                for item in value:  
                    text += f"- {item}\n"
            else:
                text += f"\n{key}:\n{value}\n"
        return text


    # collection_child and collection_parent both are chroma data bases
    def get_answer(query, collection_child, collection_parent, child_top_k=5, parent_top_k=4):
        result = collection_child.query(query_texts=[query], n_results=child_top_k, include=["documents", 'distances',]) #the way we query in chromadb
        top_childs = result['documents'][0]
        top_res = result['documents'][0]
        top_parents = []

        for ids in result['ids'][0]:
            document = collection_parent.get(qa_processing.child_id_to_parent_id(ids, 4))['documents'][0]
            top_parents.append(document)

        query_embedding = embedder.encode(query, convert_to_tensor=True)
        top_parents_embeddings = embedder.encode(top_parents, convert_to_tensor=True)  
        cos_scores = util.cos_sim(query_embedding, top_parents_embeddings)[0]  #finding the top of the top parents

        try:
            top_results_p = torch.topk(cos_scores, k=parent_top_k)
        except:
            top_results_p = torch.topk(cos_scores, k = len(top_parents_embeddings))

        updated_top_parents = []
        for i in top_results_p[1]:
            top_res.append(top_parents[i])
            updated_top_parents.append(top_parents[i])

        #answer = gpt_chunking.get_completion(qa_processing.make_categorize_conversation_prompt(query, top_res))
        answer = local_llm.get_answer(qa_processing.make_categorize_conversation_prompt(query, top_res))
        
        return answer, top_childs, updated_top_parents

    def vector_db(child_name, parent_name, child_chunk, parent_chunk, db_path):
        client = chromadb.PersistentClient(path=db_path)
        collection_child = client.create_collection(child_name)
        collection_parent = client.create_collection(parent_name)

        collection_child.add(documents = child_chunk,
        ids = list(map(lambda tup: f"id{tup[0]}", enumerate(child_chunk))))

        collection_parent.add(documents = parent_chunk,
        ids = list(map(lambda tup: f"id{tup[0]}", enumerate(parent_chunk))))


    def immport_db(child_name, parent_name, db_path):
        client = chromadb.PersistentClient(path=db_path)
        collection_child = client.get_collection(child_name)
        collection_parent = client.get_collection(parent_name)

        return collection_child, collection_parent

    def streamlit_process(query, text, db_path, child_name, parent_name):
        parent_chunk, child_chunk = gpt_chunking.process(text=text)
        qa_processing.vector_db(child_name, parent_name, child_chunk, parent_chunk, db_path)
        collection_child, collection_parent = qa_processing.immport_db(child_name, parent_name, db_path)
        answer, top_childs, updated_top_parents = qa_processing.get_answer(query, collection_child, collection_parent, child_top_k=5, parent_top_k=4)

        return answer
    def chunk_and_db(text, child_name, parent_name, db_path):
        parent_chunk, child_chunk = gpt_chunking.process(text=text)
        try:
            qa_processing.vector_db(child_name, parent_name, child_chunk, parent_chunk, db_path)
        except:
            client = chromadb.PersistentClient(path=db_path)

            client.delete_collection(child_name)
            client.delete_collection(parent_name)

            child_chunk_split = qa_processing.split_list(child_chunk, 140)
            parent_chunk_split = qa_processing.split_list(parent_chunk, 140)
            
            collection_child = client.create_collection(child_name)
            collection_parent = client.create_collection(parent_name)

            index = 0
            for splitted_childs in child_chunk_split:
                
                collection_child.add(documents = splitted_childs,
                    ids = ['id'+str(i) for i in range(index, index+len(splitted_childs))])
                index = index+len(splitted_childs)

            index = 0
            for splitted_parents in parent_chunk_split:
                
                collection_parent.add(documents = splitted_parents,
                    ids = ['id'+str(i) for i in range(index, index+len(splitted_parents))])
                index = index+len(splitted_parents)