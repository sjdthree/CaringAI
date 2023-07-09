# https://www.haihai.ai/gpt-gdrive/

# Google Drive + ChatGPT + LangChain + Python

# pip install openai
# pip install langchain
# pip install chroma
# pip install tiktoken
# pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib

from flask import Flask, render_template, request, session
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # replace with your secret key

embeddings = OpenAIEmbeddings()

with open('all_texts.pkl', 'rb') as f:
    all_texts = pickle.load(f)

# load from disk
# db = Chroma(embeddings=embeddings,persist_directory="./chroma_db")
db = Chroma.from_documents(all_texts, embeddings,  persist_directory="./chroma_db")

retriever = db.as_retriever()

print(retriever)

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        query = request.form.get('query')
        answer = qa.run(query)
        if 'chat_log' not in session:
            session['chat_log'] = []
        session['chat_log'].append({'query': query, 'answer': answer})
    return render_template('home.html', chat_log=session.get('chat_log'))

if __name__ == '__main__':
    app.run(debug=True)