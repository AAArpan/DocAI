import streamlit as st
# from dotenv import load_dotenv
from PyPDF2 import PdfReader
import re
import chromadb.utils.embedding_functions as embedding_functions
from chromadb import Client, Settings
from typing import List, Dict
import os
# import openai
from llama_cpp import Llama
from htmlTemplate import css, bot_template, user_template

load_dotenv()
Okey = os.environ.get('OPENAI_API_KEY')

# openai.api_key = Okey
Hkey = "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"

huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
api_key=Hkey,
model_name="hkunlp/instructor-xl"
)

client1 = Client(settings = Settings(persist_directory="./", is_persistent=True))
collection1_ = client1.get_or_create_collection(name="shiv", embedding_function=huggingface_ef)

def get_pdf_text(pdf_docs):
    text=""
    docs= {}
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_no in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_no]
            text +=page.extract_text()
            Split = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
            text_chunks = Split.split_text(text)
            docs[page_no] = text_chunks
    return docs, text



def add_text_to_collection(file):
    docs, text = get_pdf_text(file)
    docs_strings = [] 
    ids = []  
    metadatas = []  
    id = 0 
    
    for page_no in docs.keys():
        for doc in docs[page_no]:
            docs_strings.append(doc)
            
            metadatas.append({'page_no': page_no})
            ids.append(id)
            id += 1


    collection1_.add(
        ids=[str(id) for id in ids], 
        documents=docs_strings,  
        metadatas=metadatas,  
    )
    
    # Return a success message
    return "PDF embeddings successfully added to collection"

def query_collection(texts: str, n: int) -> List[str]:
    result = collection1_.query(
                  query_texts = texts,
                  n_results = n,
                 )
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    resulting_strings = []
    for page_no, text_chunk in zip(metadatas, documents):
        resulting_strings.append(f"Page {page_no['page_no']}: {text_chunk}")
    return resulting_strings


def get_response(queried_texts: List[str],) -> List[dict]:
    global messages
    messages = [
                {"role": "system", "content": "You are a helpful assistant.\
                 And will always answer the question asked in 'ques:' and \
                 will quote the page number while answering any questions,\
                 It is always at the start of the prompt in the format 'page n'."},
                {"role": "user", "content": ''.join(queried_texts)}
          ]

    
    llm = Llama(model_path=r"C:\Users\arpan\OneDrive\Desktop\ChatPDF\model\mistral-7b-instruct-v0.2.Q5_K_M.gguf", chat_format="llama-2")  # Set chat_format according to the model
    llama_response = llm.create_chat_completion(
        messages = [
            {"role": "system", "content": "You are a helpful assistant.\
                 And will always answer the question asked in 'ques:' and \
                 will quote the page number while answering any questions,\
                 It is always at the start of the prompt in the format 'page n'."},
            {
                "role": "user", "content": ''.join(queried_texts)
            }
    ])

    
    # response = openai.chat.completions.create(
    #                         model = "gpt-3.5-turbo",
    #                         messages = messages,
    #                         temperature=0.2,               
    #                  )

    response_msg = llama_response['choices'][0]['message']['content']
    # messages = messages + [{"role":'assistant', 'content': response_msg}]
    return response_msg


def get_answer(query: str, n: int):
    queried_texts = query_collection(texts = query, n = n)
    queried_string = [''.join(text) for text in queried_texts]
    queried_string = queried_string[0] + f"ques: {query}"
    # print(queried_string)
    answer = get_response(queried_texts = queried_string,)
    return answer

def handle_userinput(user_question):
    
    st.write(user_template.replace("{{MSG}}",user_question), unsafe_allow_html=True)
    answer = get_answer(user_question, n=1)
    
    st.write(bot_template.replace("{{MSG}}",answer), unsafe_allow_html=True)

    return answer

class SentenceSplitter:
    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        return self._split_english_text(text)

    def _split_english_text(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' '))
        chunks, current_chunk = [], ''
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size or not current_chunk:
                current_chunk += (' ' if current_chunk else '') + sentence
            else:
                chunks.append(current_chunk)
                current_chunk = sentence
        if current_chunk:  # Add the last chunk
            chunks.append(current_chunk)

        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._handle_overlap(chunks)

        return chunks

    def _handle_overlap(self, chunks: list[str]) -> list[str]:
        overlapped_chunks = []
        for i in range(len(chunks) - 1):
            chunk = chunks[i] + ' ' + chunks[i + 1][:self.chunk_overlap]
            overlapped_chunks.append(chunk.strip())
        overlapped_chunks.append(chunks[-1])
        return overlapped_chunks


def main():
    st.set_page_config(page_title="AI For Documents", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("AI For Documents :books:")
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        response = handle_userinput(user_question)
        # print(response)

    # st.write(user_template.replace("{{MSG}}","Hello bot"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}","Hello human"), unsafe_allow_html=True)
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                docs, text = get_pdf_text(pdf_docs)
                confirm = add_text_to_collection(pdf_docs)
                st.write(confirm)

                # print(type(docs))
                # print(docs)
                # create vector store


if __name__ =="__main__":
    main()

# print("hello world")
