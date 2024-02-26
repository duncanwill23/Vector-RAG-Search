from pymongo import MongoClient
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
# import gardio as gr
# from gardio.themes.base import Base
import key_param

client = MongoClient(key_param.MONGO_URI)
dbName = "langchain_chatbot"
collectionName = "chatbot_data"
collection = client[dbName][collectionName]

loader = DirectoryLoader('./sample_files', glob="*.txt", show_progress=True)
data = loader.load()

embeddings = OpenAIEmbeddings(openai_api_key = key_param.open_api_key)

vectorStore = MongoDBAtlasVectorSearch.from_documents(data, embeddings, collection=collection)