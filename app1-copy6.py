from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
os.environ["GOOGLE_API_KEY"] = st.secrets.get("GOOGLE_API_KEY")


# Create the vector store using the specified parameters
index_name = "thanet3" 
pinecone_api_key = st.secrets.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)
huggingface_ef = HuggingFaceEmbeddings(model_name="BAAI/bge-m3",model_kwargs={"device": "cpu"})
vector_store = PineconeVectorStore(index=index, embedding=huggingface_ef)

# Returns history_retriever_chain
def get_retreiver_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b", temperature=0.3)
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation and the user's question, generate a search query to look up in my document store. The search query should focus on retrieving information specifically related to the user's question and the context of the conversation, prioritizing the retrieval of relevant documents from my document store.")

    ])
    history_retriver_chain = create_history_aware_retriever(llm, retriever, prompt)
    return history_retriver_chain

# Returns conversational rag
def get_conversational_rag(history_retriever_chain):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b", temperature=0.3)
    answer_prompt = ChatPromptTemplate.from_messages([        
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    document_chain = create_stuff_documents_chain(llm, answer_prompt)
    # create final retrieval chain
    conversational_retrieval_chain = create_retrieval_chain(history_retriever_chain, document_chain)
    return conversational_retrieval_chain

# Returns th final response
def get_response(user_input):
    history_retriever_chain = get_retreiver_chain()
    conversation_rag_chain = get_conversational_rag(history_retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response["answer"]

# Streamlit app
st.header("Chat with Knowledge Base")
chat_history = []

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="สวัสดีครับ มีอะไรให้ผมช่วยไหมครับ ?")
    ]

user_input = st.chat_input("Type your message here...")
if user_input is not None and user_input.strip() != "":
    response = get_response(user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    else:
        with st.chat_message("Human"):
            st.write(message.content)
