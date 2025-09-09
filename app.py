import streamlit as st
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
import asyncio

# Fix for "no current event loop" in Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# Function to handle user questions
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load local FAISS index
    new_db = FAISS.load_local("FAISS_INDEX", embeddings, allow_dangerous_deserialization=True)

    # Search relevant chunks
    docs = new_db.similarity_search(user_question)
    st.write("🔍 Retrieved docs:", docs[:2])  # Debug: show first 2 docs

    # Get chain and response
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    # Debug: raw response
    st.write("✅ Raw Response:", response)

    # Final output
    st.subheader("💡 Answer")
    st.write(response.get("output_text", "⚠️ No answer generated"))


# Main app function
def main():
    st.set_page_config(page_title="Chat with PDFs", layout="wide")
    st.header("📚 Chat with your PDFs using Gemini Pro")

    # Sidebar: file upload
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    # Extract text
                    raw_text = get_pdf_text(pdf_docs)

                    # Split text
                    text_chunks = get_text_chunks(raw_text)

                    # Create and save FAISS index
                    get_vector_store(text_chunks)

                    st.success("✅ Done! Now you can ask questions.")
            else:
                st.warning("Please upload at least one PDF.")

    # Main area: user input
    st.subheader("Ask a question from your PDFs")
    user_question = st.text_input("Your Question:")
    if user_question:
        user_input(user_question)


if __name__ == "__main__":
    main()
