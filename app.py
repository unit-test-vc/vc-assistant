import streamlit as st
import os
import asyncio

# Disable file watcher to avoid PyTorch conflicts
os.environ['STREAMLIT_SERVER_WATCH_CHANGES'] = 'false'

# Setup asyncio loop properly
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from supabase.client import create_client

# Set page config
st.set_page_config(page_title="VC Deal Assistant", layout="wide")

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

# Get secrets
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]
hf_api_key = st.secrets["HF_API_KEY"]

# Set up Supabase client
supabase = create_client(supabase_url, supabase_key)

# Set up embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Vector Store
@st.cache_resource
def get_vectordb():
    embeddings = get_embeddings()
    return SupabaseVectorStore(
        embedding=embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents"
    )

# Set up LLM
@st.cache_resource
def get_llm():
    return HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=hf_api_key,
        model_kwargs={"temperature": 0.7, "max_length": 1024}
    )

# Streamlit UI
st.title("VC Deal Analysis Assistant")

# Sidebar for uploading and settings
with st.sidebar:
    st.header("Upload Documents")
    uploaded_file = st.file_uploader("Upload pitch deck or document", type=["pdf", "txt", "csv"])
    
    if uploaded_file:
        if uploaded_file.name not in st.session_state.processed_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # Save uploaded file temporarily
                temp_file_path = f"temp_file.{uploaded_file.name.split('.')[-1]}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process document based on type
                if uploaded_file.name.endswith('.pdf'):
                    loader = PyPDFLoader(temp_file_path)
                elif uploaded_file.name.endswith('.csv'):
                    loader = CSVLoader(temp_file_path)
                else:
                    loader = TextLoader(temp_file_path)
                
                documents = loader.load()
                
                # Split documents
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = text_splitter.split_documents(documents)
                
              # Add to vector store
try:
    vectordb = get_vectordb()
    
    # Add metadata to identify the document
    for i, text in enumerate(texts):
        text.metadata["source"] = uploaded_file.name
        text.metadata["chunk_id"] = i
    
    # Process in smaller batches to avoid issues
    batch_size = 5
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            vectordb.add_documents(batch)
            st.sidebar.write(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        except Exception as e:
            st.sidebar.error(f"Error in batch {i//batch_size + 1}: {str(e)}")
            continue
    
    # Add to processed files
    st.session_state.processed_files.append(uploaded_file.name)
    st.success(f"Added {uploaded_file.name} to knowledge base!")
except Exception as e:
    st.error(f"Error adding documents to database: {str(e)}")
    st.info("Try re-configuring the database or using smaller documents.")
                
    # Clean up temp file
        os.remove(temp_file_path)
                
                st.success(f"Added {uploaded_file.name} to knowledge base!")
        else:
            st.info(f"{uploaded_file.name} has already been processed.")
    
    st.header("Knowledge Base")
    if st.session_state.processed_files:
        st.write("Processed documents:")
        for file in st.session_state.processed_files:
            st.write(f"- {file}")
    else:
        st.write("No documents processed yet.")

# Main area for querying
st.header("Ask About Your Deals")
query = st.text_area("Enter your question:", height=100)

if st.button("Submit"):
    if query:
        vectordb = get_vectordb()
        
        # Check if we have documents
        try:
            retriever = vectordb.as_retriever(search_kwargs={"k": 4})
            
            with st.spinner("Analyzing..."):
                # Initialize LLM
                llm = get_llm()
                
                # Create retrieval chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
                )
                
                # Get response
                response = qa_chain.invoke(query)
                
                st.subheader("Analysis")
                st.write(response["result"])
                
                # Show sources
                st.subheader("Sources")
                sources = set()
                for doc in response["source_documents"]:
                    if "source" in doc.metadata:
                        sources.add(doc.metadata["source"])
                
                if sources:
                    st.write("Information drawn from:")
                    for source in sources:
                        st.write(f"- {source}")
        except Exception as e:
            st.error(f"Please upload some documents first or check your database connection. Error: {str(e)}")
