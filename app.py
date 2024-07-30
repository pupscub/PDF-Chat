import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS

def get_raw_text(pdfs) -> str:
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Avoid appending None
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl',model_kwargs = {'device':'cpu'})
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def main():
    st.set_page_config(
        page_icon=":books:",
        page_title="PDF Chat"
    )
    st.header("Chat with Multiple PDFs :books:")

    # Sidebar for file upload
    with st.sidebar:
        st.subheader("Your Document")
        pdfs = st.file_uploader("Upload the PDFs", type="pdf", accept_multiple_files=True)
        if st.button("Process") and pdfs:
            with st.spinner("Processing your PDFs...."):
                try:
                    raw_text = get_raw_text(pdfs)

                    if not raw_text:
                        st.error("No text extracted from the PDFs.")
                        return

                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vector_store(text_chunks)

                    st.success("Processing complete!")
                    st.text_area("Text Chunks Preview", "\n".join(text_chunks[:5]))  # Show a preview of the text chunks

                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # User input for questions
    user_question = st.text_input("Ask a question related to the document")
    if user_question:
        st.write("You asked:", user_question)
        # Integrate logic to search the vectorstore and answer the question

if __name__ == "__main__":
    main()
