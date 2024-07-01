import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv




load_dotenv()


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


st.set_page_config(
    page_title="Chat PDF APP",
    page_icon= ":filetype-pdf:",
    layout="centered"
)


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=800)
    chunks = text_splitter.split_text(text)
    return chunks
def get_vector_stores(chunks):
    embedings = GoogleGenerativeAIEmbeddings(model= "models/embedding-001")
    vector_store = FAISS.from_texts(chunks,embedding=embedings)
    vector_store.save_local('faiss_index')



def get_conversation_chain():
    template=""""
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n 
    Answer:
    """
    model = ChatGoogleGenerativeAI(model = 'gemini-1.0-pro',temperature=0.5)


    prompt = PromptTemplate(input_variables =['context','question'],template=template)
    chain = load_qa_chain(llm=model,chain_type='stuff',prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversation_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.success(response["output_text"])

st.header("Chat with PDF using Gemini💁")

user_question = st.text_input("Ask a Question from the PDF Files")

if user_question:
    user_input(user_question)

with st.sidebar:
    st.title("Menu:")
    pdf = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True,type=['pdf'])
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf)
            text_chunks = get_chunks(raw_text)
            get_vector_stores(text_chunks)
            st.success("Done")


