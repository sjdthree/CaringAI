# https://www.haihai.ai/gpt-gdrive/

# Google Drive + ChatGPT + LangChain + Python
#  change to vertexai by google instead of ChatGPT

# pip install openai

# pip install google-cloud-aiplatform

# pip install langchain
# pip install pypdf2
# pip install chroma
# pip install tiktoken
# pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib

# pip install google-cloud-aiplatform

from langchain.chat_models import ChatVertexAI


from langchain.chains import RetrievalQA
from langchain.document_loaders import GoogleDriveLoader


# from new source: How to build LLM Model app using Langchain and Google
import pandas as pd
# Utils
import time
from typing import List
from pydantic import BaseModel
from google.cloud import aiplatform
from langchain.chat_models import ChatVertexAI
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import Chroma
import whisper
from pytube import YouTube
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import vertexai
PROJECT_ID = "caringai"  # @param {type:"string"}
vertexai.init(project=PROJECT_ID, location="us-central1")

# # folder_id = "1YMWmt_We-pxDt4hvBv5I3WGJV61oaH88"
# folder_id = "1I7K05pGyPw6dE5jTke0nyH0xnICled_G"
# # This is parenting books
# loader = GoogleDriveLoader(
#     folder_id=folder_id,
#     recursive=False
# )

loader = PyPDFLoader("mentallystrong.pdf")

def rate_limit(max_per_minute):
    min_interval = 60.0 / max_per_minute  # Minimum interval between operations

    while True:
        start_time = time.time()  # Start time of operation
        yield  # Perform the operation
        end_time = time.time()  # End time of operation

        elapsed_time = end_time - start_time  # Calculate elapsed time
        sleep_time = max(0, min_interval - elapsed_time)  # Calculate sleep time if needed

        if sleep_time > 0:
            time.sleep(sleep_time)  # Sleep to maintain the rate limit


class CustomVertexAIEmbeddings(VertexAIEmbeddings, BaseModel):
    requests_per_minute: int
    num_instances_per_batch: int

    # Overriding embed_documents method
    def embed_documents(self, texts: List[str]):
        limiter = rate_limit(self.requests_per_minute)
        results = []
        docs = list(texts)

        while docs:
            # Working in batches because the API accepts maximum 5
            # documents per request to get embeddings
            head, docs = (
                docs[: self.num_instances_per_batch],
                docs[self.num_instances_per_batch :],
            )
            chunk = self.client.get_embeddings(head)
            results.extend(chunk)
            next(limiter)

        return [r.values for r in results]
        
        # Embedding
EMBEDDING_QPM = 20
EMBEDDING_NUM_BATCH = 2
embeddings = CustomVertexAIEmbeddings(
    requests_per_minute=EMBEDDING_QPM,
    num_instances_per_batch=EMBEDDING_NUM_BATCH,
)

# alternate would be:
# loader = PyPDFLoader("yourpdf.pdf")

docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000, chunk_overlap=0, separators=[" ", ",", "\n"]
    )

texts = text_splitter.split_documents(docs)

db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever()


llm = ChatVertexAI()
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)

# qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

from langchain.schema import (
    HumanMessage
)
# prepare template for prompt
template = """You are an advanced AI assistant that summarizes online articles into bulleted lists.

Here's the article you need to summarize.

==================
Title: {article_title}

{article_text}
==================

Now, provide a summarized version of the article in a bulleted list format.
"""

# format prompt
prompt = template.format(article_title="Activation functions in JAX and Flax", article_text="texts")

# generate summary
summary = llm([HumanMessage(content=prompt)])
print(summary.content)


while True:
    query = input("> ")
    answer = qa.run(query)
    print(answer)