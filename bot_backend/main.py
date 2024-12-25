from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import logging
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

app = FastAPI()

# Configure LangSmith (if applicable)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Zain GPT"
os.environ["LANGCHAIN_API_KEY"] = ""
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBv9RRwN5FJMXCM3-UvNQSz1x584yfKl3o")

# Define the input model
class Question(BaseModel):
    question: str

# Define the separators (sections) within the text file
separators = ["Personal Information", "Education", "Technical Skills", "Degrees/Certifications/Courses:", "Projects :", "Experience:"]

# Function to read the text from a file
def get_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Function to split the text into chunks
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50, separators=separators)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save vector store with embeddings
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=GOOGLE_API_KEY)
    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local('chroma_index')  # Save vector store locally

# Function to create the QA chain
def get_chain():
    prompt_template = """You are Zain GPT, an intelligent assistant created to answer all questions related to Zain Attiq, a Cloud Applied Generative AI Engineer. Your goal is to provide clear, well-organized, and human-like responses based on the Context and Conversation History provided below.
    Must pay attention to the Conversation history to understand well what is the state of the conversation and try to carry it on.\n

    Please keep the following guidelines in mind:\n
    - Respond to greetings with a polite and warm tone and introduce yourself and tell them what you can assist them with.
    - Structure your answers using bullet points, tables (where applicable), and always include a conversational opening sentence to make your responses feel more natural and human-like.
    - When answering questions about Zain Attiq, be detailed, polite, and sweet in your tone.
    - Your scope is limited to answering questions about Zain Attiq and things related to him. If a user asks something unrelated, kindly inform them that you can only provide answers related to Zain Attiq.
    - For project-related queries, provide the project names along with their respective links and video presentation links from the provided context.
    - If you cannot find the answer within the provided context, respond with: "Sorry, I can't answer this right now. You can contact Zain Attiq at zainatteeq@gmail.com." Never give incorrect information.
    - End every response with: "Is there anything else you would like to know about Mr. Zain Attiq?"

    \n\n
    Context: \n{context}\n
    User Query: \n{question}\n
    Conversation History: \n{history}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "history"])
    chain = load_qa_chain(llm=model, chain_type='stuff', prompt=prompt)
    return chain 

# FastAPI endpoint to process user queries
@app.post("/ask")
async def ask_question(question: Question):
    user_question = question.question

    # Initialize the session history if not already present
    if "history" not in st.session_state:
        st.session_state["history"] = []
        st.session_state["history"].append(SystemMessage("You are Zain GPT. You will answer all the queries related to him. Zain is a Cloud Applied Generative AI with a diverse skillset. He is still learning state-of-the-art technologies and working hard on his skills. He has lofty ambitions and wants to make a difference in unique ways.")) 

    st.session_state["history"].append(HumanMessage(user_question))

    # Specify the file path and read the text
    text_path = r"F:\Portfolio\bot-backend\bot_backend\zaingpt.txt"  # Replace with your actual path
    raw_text = get_text(text_path)
    text_chunks = get_chunks(raw_text)

    # Create embeddings and search for relevant documents
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=GOOGLE_API_KEY)
    new_db = Chroma.from_texts(text_chunks, embedding=embeddings)
    
    docs = new_db.similarity_search(user_question)
    
    # If no documents found, raise an exception
    if not docs:
        raise HTTPException(status_code=404, detail="No relevant context found for the question.")

    # Get the QA chain and process the question
    chain = get_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question, "history": st.session_state["history"]}, return_only_outputs=True)
    
    # Append the response to the conversation history
    st.session_state["history"].append(AIMessage(response["output_text"] + "\n"))

    return {"response": response["output_text"]}

# Function to initialize the vector store (if not already initialized)
if __name__ == "__main__":
    text_path = "F:/Portfolio/bot-backend/bot_backend/zaingpt.txt"  # Replace with your actual path

    # Process the text file and create embeddings once
    if not os.path.exists("chroma_index"):
        raw_text = get_text(text_path)
        if raw_text:
            text_chunks = get_chunks(raw_text)
            if text_chunks:
                get_vector_store(text_chunks)

# Starting the FastAPI application using Uvicorn
def start():
    import uvicorn
    uvicorn.run("bot_backend.main:app", host="127.0.0.1", port=8080, reload=True)
