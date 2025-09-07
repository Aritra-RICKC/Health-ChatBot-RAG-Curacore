import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables from a .env file
load_dotenv(find_dotenv())

# --- CONFIGURATION ---
DB_FAISS_PATH = "vectorstore/db_faiss"
MODEL_NAME = "llama-3.1-8b-instant" # You can change this to other models like "mixtral-8x7b-32768"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

# --- CACHED FUNCTIONS ---

@st.cache_resource
def get_vectorstore():
    """Loads the FAISS vector store from the local path."""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
        return None

@st.cache_resource
def get_qa_chain():
    """Initializes and returns the RetrievalQA chain."""
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return None

    try:
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ_API_KEY is not set. Please add it to your .env file.")
            st.stop()
            
        llm = ChatGroq(
            model_name=MODEL_NAME,
            temperature=0.5, # Lower temperature for more factual answers
            api_key=groq_api_key,
        )

        prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': prompt}
        )
        return qa_chain
    except Exception as e:
        st.error(f"Failed to create QA chain: {e}")
        return None


# --- MAIN APPLICATION LOGIC ---
def main():
    st.set_page_config(page_title="Chat with Your Data", page_icon="🤖")
    st.title("Ask Your Healthbot!")

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Get the QA chain
    qa_chain = get_qa_chain()

    # Handle user input
    if prompt := st.chat_input("Ask your issue..."):
        # Add user message to chat history
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)

        # Process the prompt if the chain is available
        if qa_chain:
            try:
                with st.spinner("Thinking..."):
                    response = qa_chain.invoke({'query': prompt})
                    result = response.get("result", "Sorry, I couldn't find an answer.")
                    source_documents = response.get("source_documents", [])

                # Display assistant response
                with st.chat_message('assistant'):
                    st.markdown(result)
                    if source_documents:
                        with st.expander("View Sources"):
                            for doc in source_documents:
                                st.markdown(f"**Source:** `{doc.metadata.get('source', 'N/A')}`")
                                st.markdown(f"> {doc.page_content}")
                
                # Add assistant response to chat history
                st.session_state.messages.append({'role': 'assistant', 'content': result})

            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({'role': 'assistant', 'content': error_message})
        else:
            st.error("The question answering chain is not available. Please check the logs.")


if __name__ == "__main__":
    main()
