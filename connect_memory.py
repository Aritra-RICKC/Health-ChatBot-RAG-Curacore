import os

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

def load_llm(model_name):
    """Initializes and returns the ChatGroq LLM instance."""
    llm = ChatGroq(
        model_name=model_name,
        temperature=0.8,
        api_key=GROQ_API_KEY
    )
    return llm

# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    """
    Sets up a custom prompt template for the QA chain.
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Step 3: Load Database and Create QA Chain

# Define the path for the local FAISS vector store
DB_FAISS_PATH = "vectorstore/db_faiss"

# Initialize the embedding model from Hugging Face
# This part remains the same as we are still using the same embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the FAISS database from the local path
# The allow_dangerous_deserialization is necessary for loading FAISS indexes created with older versions.
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(GROQ_MODEL_NAME),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 4: Invoke the chain with a user query
if __name__ == "__main__":
    user_query = input("Write Query Here: ")
    response = qa_chain.invoke({'query': user_query})
    print("\nRESULT: ", response["result"])
    print("\nSOURCE DOCUMENTS: ")
    for doc in response["source_documents"]:
        print(f"- {doc.page_content}")