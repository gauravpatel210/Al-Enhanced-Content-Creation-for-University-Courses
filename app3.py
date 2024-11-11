import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load API keys from environment
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, just say,
    'answer is not available in the context'. In either case, generate a plagiarism-free response using the information or general knowledge.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain
def generate_plagiarism_free_content(context):
    content_generation_prompt = """
    Generate original content based on the following context. Ensure that the content is unique and does not replicate the provided text verbatim.
    Context:\n{context}\n
    Generated Content:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    prompt = PromptTemplate(template=content_generation_prompt, input_variables=["context"])
    
    # Create a document-like structure
    input_docs = [{"page_content": context}]
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    # Pass the input_documents key with a list of documents
    response = chain({"input_documents": input_docs}, return_only_outputs=True)
    return response["output_text"]



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    if docs:
        # If there is relevant context, use it to generate an answer
        response = chain({"input_documents": docs, "question": user_question})
        generated_content = generate_plagiarism_free_content(response["output_text"])
    else:
        # If no context is found, generate a plagiarism-free response using general knowledge
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
        general_knowledge_response = model(user_question)
        generated_content = generate_plagiarism_free_content(general_knowledge_response["output_text"])

    st.write("Reply: ", generated_content)

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using Gemini")

    user_question = st.text_input("Ask a Question from the PDF Files or General Knowledge")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

                # Generate plagiarism-free content from the processed text
                generated_content = generate_plagiarism_free_content(raw_text)
                st.write("Generated Plagiarism-Free Content: ", generated_content)

if __name__ == "__main__":
    main()
