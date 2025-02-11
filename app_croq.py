# from langchain.chains import RetrievalQA
import os
from PyPDF2 import PdfReader
from langchain.schema import Document
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from typing_extensions import Concatenate
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

if "PINECONE_API_KEY" not in os.environ:
    raise ValueError("Please set PINECONE_API_KEY in your .env file")
if "GROQ_API_KEY" not in os.environ:
    raise ValueError("Please set GROQ_API_KEY in your .env file")

# Initialize Pinecone
pc = Pinecone(
    api_key=os.environ["PINECONE_API_KEY"]
)
pdf_path = r"sample.pdf"
# Read PDF
pdf_reader = PdfReader(pdf_path)

# Extract text from PDF
raw_text = ""
for i,page in enumerate(pdf_reader.pages):
    content = page.extract_text()
    if content:
        raw_text += content
# print(raw_text)

# Create document
docs = [Document(page_content=raw_text)]

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    length_function=len,
)
texts = text_splitter.split_documents(docs)

# Create metadata with source information
documents_with_sources = []
web_url = "web_url"
for doc in texts:
    doc.metadata = {
        "source": doc.metadata.get("source", "Course_Name"),  # Original PDF path
        "source_url": web_url,  # Clickable file URL
        "text": doc.page_content,
    }
    documents_with_sources.append(doc)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")

# Create or get existing index
index_name = "course-database"
existing_indexes = pc.list_indexes().names()
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)

# Initialize Pinecone vectorstore
vectorstore = LangchainPinecone.from_documents(
    documents_with_sources,
    embedding=embeddings,
    index_name=index_name
)

print("Embeddings added to Pinecone")

# # Initialize LLM with increased max_tokens
# llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=4000, temperature=0.3)

# # Create QA chain
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vectorstore.as_retriever()
# )

# # Example query
# query = "courses in mechanical semester 1?"
# result = qa.invoke(query)
# print(result)