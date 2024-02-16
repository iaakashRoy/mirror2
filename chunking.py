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

stream = False

url = "https://chat.tune.app/api/chat/completions"
headers = {
    "Authorization": "tune-b4042fc3-b3ae-4b05-a24e-b26dc3b2c0241708053579",
    "Content-Type": "application/json"
}

class gpt_chunking:       
    
    def get_completion(query):
        data = {
            "temperature": 0.5,
            "messages": [
                {
                    "role": "system",
                    "content": "You are assistant of an ai company named 'mirror2', here to help anyone with their personal documents"
                },
                {
                    "role": "user",
                    "content": "Act like, you're the assistant smart, genuine person"
                }
            ],
            "model": "mixtral-8x7b-inst-v0-1-32k",
            "stream": stream,
            "max_tokens": 300
        }
        response =  requests.post(url, headers=headers, json=data).json()
        response = response['choices'][0]['message']
        return response ['content']

    #Function to get the chat completion
    # def get_completion(prompt, model="gpt-4-8k-0613"):
    #     messages = [{"role": "user", "content": prompt}]
    #     response = openai.ChatCompletion.create(
    #         model=model,
    #         messages=messages,
    #         temperature=0, # this is the degree of randomness of the model's output
    #     )
    #     response = json.loads(response)
    #     #return response
    #     return response["choices"][0]["message"]["content"]
    
    #Function for creation of the instruction to extract rhe headings and the informationn according to the available data
    """text: Introduction \\n This part defines the transformer architecture. \\n 1.1 Encoder \\n This is the application of the multi headed system.... \\n Decoder\\n ...
       output: {'Introduction':[' This part defines the transformer architecture.', {'Encoder' : ['This is the application of the multi headed system....'], 'Decoder': ['...']}]}
    """


    #One-shot
    def get_headings_prompt(tex):
        string = {'Model Architecture': ['Most competitive neural sequence transduction models have an encoder-decoder structure [ 5,2,29].\nHere, the encoder maps an input sequence of symbol representations (x1;:::;x n)to a sequence\nof continuous representations z= (z1;:::;z n). Given z, the decoder then generates an output\nsequence (y1;:::;y m)of symbols one element at a time. At each step the model is auto-regressive\n[9], consuming the previously generated symbols as additional input when generating the next.\nThe Transformer follows this overall architecture using stacked self-attention and point-wise, fully\nconnected layers for both the encoder and decoder, shown in the left and right halves of Figure 1,\nrespectively.', {'Encoder and Decoder Stacks': ['', {'Encoder':['The encoder is composed of a stack of N= 6 identical layers. Each layer has two\nsub-layers. The ﬁrst is a multi-head self-attention mechanism, and the second is a simple, position-\n2Figure 1: The Transformer - model architecture.\nwise fully connected feed-forward network. We employ a residual connection [ 10] around each of\nthe two sub-layers, followed by layer normalization [ 1]. That is, the outputof each sub-layer is\nLayerNorm( x+ Sublayer( x)), where Sublayer(x)is the function implemented by the sub-layer\nitself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding\nlayers, produce outputs of dimension dmodel = 512 .', {}], 'Decoder':[' The decoder is also composed of a stack of N= 6identical layers. In addition to the two\nsub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head\nattention over the output of the encoder stack. Similar to the encoder, we employ residual connections\naround each of the sub-layers, followed by layer normalization. We also modify the self-attention\nsub-layer in the decoder stack to prevent positions from attending to subsequent positions.', {}]}]}]}
        instruction = f"""
        You will be given some text and you have to figure out the headings and sub-headings from the given text as follows: //
        Example: //
        text_1: \n3 Model Architecture\nMost competitive neural sequence transduction models have an encoder-decoder structure [ 5,2,29].\nHere, the encoder maps an input sequence of symbol representations (x1;:::;x n)to a sequence\nof continuous representations z= (z1;:::;z n). Given z, the decoder then generates an output\nsequence (y1;:::;y m)of symbols one element at a time. At each step the model is auto-regressive\n[9], consuming the previously generated symbols as additional input when generating the next.\nThe Transformer follows this overall architecture using stacked self-attention and point-wise, fully\nconnected layers for both the encoder and decoder, shown in the left and right halves of Figure 1,\nrespectively.\n3.1 Encoder and Decoder Stacks\nEncoder: The encoder is composed of a stack of N= 6 identical layers. Each layer has two\nsub-layers. The ﬁrst is a multi-head self-attention mechanism, and the second is a simple, position-\n2Figure 1: The Transformer - model architecture.\nwise fully connected feed-forward network. We employ a residual connection [ 10] around each of\nthe two sub-layers, followed by layer normalization [ 1]. That is, the output of each sub-layer is\nLayerNorm( x+ Sublayer( x)), where Sublayer(x)is the function implemented by the sub-layer\nitself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding\nlayers, produce outputs of dimension dmodel = 512 .\nDecoder: The decoder is also composed of a stack of N= 6identical layers. In addition to the two\nsub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head\nattention over the output of the encoder stack. Similar to the encoder, we employ residual connections\naround each of the sub-layers, followed by layer normalization. We also modify the self-attention\nsub-layer in the decoder stack to prevent positions from attending to subsequent positions. //
        output:  {string} //
        Now comes the real task! //
        The given text is : {tex} //
        Give the output for the above text as per the example. //
        DO NOT add anything extra or modify the text, search the headings and sub-headings from the given text and add accordingly//
        Must return the output in the python dictionary format only.
        """

        return instruction

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
    
    def organise(text):
        
        #return gpt_chunking.get_completion(gpt_chunking.get_headings_prompt(text))
        return local_llm.get_answer(gpt_chunking.get_headings_prompt(text))
    
    def process(text='', parent_size = 6, num_childs = 3, cores=mp.cpu_count(), batch_size = 10):
        
        #text pre-processing
        text = text.replace('\n', ' ')
        text = text.replace('................................', ' ')

        #creating the parent_child dataframe
        chunked_df = pd.DataFrame()
        child_size = parent_size//num_childs 
        total_sent = len(text.split(". "))

        chunked_df = pd.DataFrame()
        chunked_df['Parent_data'] = ''
        chunked_df['Child_data1'] = ''
        chunked_df['Child_data2'] = ''
        chunked_df['Child_data3'] = ''
        #chunked_df['Child_data4'] = ''
        i = 0

        #Improve this slicing using LLM
        for index in range(0, total_sent, parent_size): 
            end_range = min(index + parent_size , total_sent)
            chunked_df.loc[i, 'Parent_data'] = gpt_chunking.slice_text(text, index, end_range)
            chunked_df.loc[i, 'Child_data1'] = gpt_chunking.slice_text(text, index, int(min(index + child_size , total_sent)))
            chunked_df.loc[i, 'Child_data2'] = gpt_chunking.slice_text(text, int(min(index + child_size , total_sent)), int(min(index + 2*child_size , total_sent)))
            chunked_df.loc[i, 'Child_data3'] = gpt_chunking.slice_text(text, int(min(index + 2*child_size , total_sent)), int(min(index + 3*child_size , total_sent)))
            #chunked_df.loc[i, 'Child_data4'] = gpt_chunking.slice_text(text, int(min(index + 3*child_size , total_sent)), end_range)
            i += 1

        pool = ThreadPool(cores)

        total_rows = len(chunked_df['Parent_data'])  # Total number of rows in the dataframe

        #organising the chunks into the dictionary format->improve it using LLM to organise information in a more better way
        for index in stqdm(range(0, total_rows, batch_size)):
            end_range = min(index + batch_size , total_rows) 
            data = chunked_df.iloc[index:end_range]
            data['Parent_data'] = pool.map(gpt_chunking.organise, data['Parent_data'])
            data['Child_data1'] = pool.map(gpt_chunking.organise, data['Child_data1'])
            data['Child_data2'] = pool.map(gpt_chunking.organise, data['Child_data2'])
            data['Child_data3'] = pool.map(gpt_chunking.organise, data['Child_data3'])
            #data['Child_data4'] = pool.map(gpt_chunking.organise, data['Child_data4'])

            if index == 0:
                df_processed = data.copy()
            else:  
                df_processed = pd.concat([df_processed, data], axis=0) 
                
        parent_chunk = list(df_processed['Parent_data'])

        child_chunk = []
        for i in range(len(df_processed['Child_data1'])):
            child_chunk.append(df_processed['Child_data1'][i])
            child_chunk.append(df_processed['Child_data2'][i])
            child_chunk.append(df_processed['Child_data3'][i])
            #child_chunk.append(df_processed['Child_data4'][i])

        return parent_chunk, child_chunk

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