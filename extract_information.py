from pymongo import MongoClient
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
import key_param

client = MongoClient(key_param.MONGO_URI)
dbName = "langchain_chatbot"
collectionName = "chatbot_data"
collection = client[dbName][collectionName]

embeddings = OpenAIEmbeddings(openai_api_key = key_param.open_api_key)

vectprStore = MongoDBAtlasVectorSearch( collection, embeddings)

def query_data(query):
    docs = vectprStore.similarity_search(query, k=1)
    if not docs:
        return "Sorry, I dont know that. Please ask a doctor or a professional.", None
    as_output = docs[0].page_content
    llm = OpenAI(openai_api_key = key_param.open_api_key, temperature=0)
    retriever = vectprStore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
    retriever_output = qa.run(query)
    return as_output, retriever_output
    
with gr.Blocks(theme=Base(), title="Chatbot") as chatbot:
    gr.Markdown("Welcome to the chatbot! Ask me anything.")
    textbox = gr.Textbox("Ask me anything")
    with gr.Row():
        button = gr.Button("Submit", variant="primary")
    with gr.Column():
        output1 = gr.Textbox("Answer from the vector store")
        output2 = gr.Textbox("Answer from the retriever")
    button.click(query_data, textbox, outputs=[output1, output2])

chatbot.launch()
