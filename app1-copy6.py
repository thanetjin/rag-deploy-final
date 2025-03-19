from langchain_core.messages import HumanMessage, AIMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import io
import re
import streamlit as st
import os
import tempfile
import time
from PyPDF2 import PdfReader
from langchain.schema import Document

os.environ["LLAMA_CLOUD_API_KEY"] = st.secrets.get("LLAMA_CLOUD_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
import nest_asyncio
nest_asyncio.apply()

load_dotenv()

# Load API Keys

pinecone_api_key = st.secrets.get("pinecone_api_key")
# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index_list = pc.list_indexes()
index_names = [index_info["name"] for index_info in index_list.get("indexes", [])]
print("Index names are:", index_names[0])

index = pc.Index(index_names[0])
huggingface_ef = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vector_store = PineconeVectorStore(index=index, embedding=huggingface_ef)

def handle_upload():
    # This is triggered when a file is uploaded
    st.session_state.file_uploaded = True

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="สวัสดีครับผมครูเก่ง 👨🏻‍💻 สามารถสอบถามข้อมูลเกี่ยวกับหลักสูตรวิทยาการคอมพิวเตอร์และนโยบายเกี่ยวกับการเรียนได้ในมหาวิทยาลัยเกษตรศาสตร์นะครับ")
    ]
if "last_interaction" not in st.session_state:
    st.session_state.last_interaction = time.time()
    # Global flag to track deletio
if "deletion_scheduled" not in st.session_state:
    st.session_state.deletion_scheduled = False
if "student_id" not in st.session_state:    
    st.session_state["student_id"] = None  # or some default value
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = None
if "is_transcript_uploaded" not in st.session_state:
    st.session_state.is_transcript_uploaded = False  # Controls if a valid transcript was uploaded
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False  # False means user has not uploaded yet
if "temp_transcript_data" not in st.session_state:
    st.session_state.temp_transcript_data = None
st.header("💬 Chat with Kru Keng")
st.title("👨‍🏫 Ask me anything about the Computer Science curriculum at Kasetsart University or academic policies and procedures")
uploaded_file = st.file_uploader(
    "Upload your transcript", 
    type=["pdf"], 
    disabled=st.session_state.is_transcript_uploaded,  # Disable only if a transcript is confirmed
    on_change=handle_upload
)

def get_retriever_chain():
    """Retrieve documents based on user queries, distinguishing between course and transcript data."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")         
    metadata_filter = {
        "$or": [            
            {"type": "course"},
            {"type": "english"},
            {"type": "ged-ed"},
            {"type": "policy"}
        ]
    }
    retriever = vector_store.as_retriever(search_kwargs={'k': 100, 'filter': metadata_filter})

    prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    (
        "user",
        "วิเคราะห์คำถามของผู้ใช้และจำแนกให้อยู่ในหนึ่งในหมวดหมู่ต่อไปนี้: 'ข้อมูลรายวิชา', "
        "'ผลการเรียนของนักศึกษา', หรือ 'ทั้งสองอย่าง' จากนั้นสร้างคำค้นหาที่เกี่ยวข้องมากที่สุด\n\n"
        "**สำคัญ:**\n"
        "- หากผู้ใช้สอบถามเกี่ยวกับคำแนะนำรายวิชา ตรวจสอบให้แน่ใจว่ารายวิชาที่ลงทะเบียนไปแล้วจะไม่ถูกแนะนำซ้ำ\n"
        "- ตรวจสอบผลการเรียนของนักศึกษาก่อน (หากมี) ก่อนแนะนำรายวิชาใหม่\n"
        "- ดึงข้อมูลที่เกี่ยวข้องที่สุดเท่านั้น เพื่อลดผลลัพธ์ที่ไม่จำเป็น"
    )
])
    history_retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return history_retriever_chain

def get_conversational_rag(history_retriever_chain):
    """Creates a RAG chain that dynamically retrieves course and transcript information."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        คุณคือผู้ช่วย AI ที่ช่วยนักศึกษาเกี่ยวกับการลงทะเบียนเรียนและนโยบายทางวิชาการ
        เป้าหมายของคุณคือการให้คำตอบที่ชัดเจนและกระชับโดยอิงจากคำถามของนักศึกษาโดยตรง
        ใช้เฉพาะข้อมูลที่เกี่ยวข้องและหลีกเลี่ยงรายละเอียดที่ไม่จำเป็น

        **แนวทางการตอบคำถาม:**
        - ตอบคำถามอย่างกระชับและตรงประเด็น
        - หากคำถามไม่ชัดเจน ให้ขอรายละเอียดเพิ่มเติมจากนักศึกษา
        - หากไม่มีข้อมูล ให้แนะนำให้นักศึกษาติดต่อสำนักงานทะเบียนหรือฝ่ายวิชาการ
        - นักเรียนสามารถอัพโหลด transcirpt (ใบเกรด) จงใช้ข้อมูลจากที่นักเรียนอัพโหลดในการตอบคำถามด้วยหากเกี่ยวข้องกับ transcript นักเรียนจะได้สะดวกไม่จำเป็นต้องเขียนรายวิชาเพิ่มสอบถาม
        - หากใน knowledgebase ไม่มีข้อมูล คุณสามารถสืบค้นข้อมูลเกี่ยวกับมหาวิทยาลัยเกษตรศาตร์ ประเทศไทย ได้ 

        **ข้อมูลที่เกี่ยวข้อง:**
        {context}
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    document_chain = create_stuff_documents_chain(llm, answer_prompt)
    conversational_retrieval_chain = create_retrieval_chain(history_retriever_chain, document_chain)
    return conversational_retrieval_chain

# Returns the final response
def get_response(user_input, extracted_text):
    """This function usesต the extracted text from the PDF directly in the prompt."""
    history_retriever_chain = get_retriever_chain()
    conversation_rag_chain = get_conversational_rag(history_retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input,        
    })

    return response["answer"]

# Function to extract Student ID dynamically
def extract_student_id(text):
    match = re.search(r"Student\s*No[:\s]*([\d]+)", text, re.IGNORECASE)  
    return match.group(1) if match else None

# Add this to your session state initialization
if "temp_transcript_data" not in st.session_state:
    st.session_state.temp_transcript_data = None

if uploaded_file and st.session_state.file_uploaded:  
    print("uploaded_file is : ",uploaded_file)
    print("st.session_state.file_uploaded : ",st.session_state.file_uploaded)
    try:
        with st.spinner("Processing PDF..."):
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_file_path = temp_file.name

            # Extract text from PDF
            extracted_text = ""
            with open(temp_file_path, "rb") as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    extracted_text += page.extract_text() + "\n"            
            is_transcript = "Date Of Graduation" in extracted_text
            
            if is_transcript:
                st.success("The uploaded document is identified as a transcript.")                
                student_id = extract_student_id(extracted_text)
                st.session_state["student_id"] = student_id  # Store it for later use            
                st.session_state["extracted_text"] = extracted_text  # Store the extracted text                                    

                # ✅ Store the transcript message but disable further uploads
                message = HumanMessage(content=extracted_text)  
                st.session_state.chat_history.append(message)                
                st.session_state.is_transcript_uploaded = True  # ✅ Prevent further uploads
            else:                
                st.error("⚠️ The uploaded document does not appear to be a transcript. Please re-upload.")
                st.session_state.file_uploaded = False  # ✅ Allow re-uploading

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
user_input = st.chat_input("Type your message here...")
if user_input and user_input.strip():
    # Use the extracted text from session state if available
    transcript_text = st.session_state.get("extracted_text")
    if transcript_text:
        print("extracted_text สำเร็จ:", transcript_text[:100])  # Print first 100 chars
    else:
        print("No extracted text found in session state")
    response = get_response(user_input, extracted_text=transcript_text)
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))
    st.session_state.last_interaction = time.time()

# ✅ Display chat history but SKIP the transcript upload message
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage) and message.content == st.session_state.get("extracted_text"):
        continue  # Skip displaying the transcript message
    with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
        st.write(message.content)
