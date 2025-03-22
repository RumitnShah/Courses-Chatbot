import os
from PyPDF2 import PdfReader
from langchain.schema import Document
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()

# Ensure necessary API keys are set
if "PINECONE_API_KEY" not in os.environ:
    raise ValueError("Please set PINECONE_API_KEY in your .env file")
if "GROQ_API_KEY" not in os.environ:
    raise ValueError("Please set GROQ_API_KEY in your .env file")

# Initialize Pinecone client
pc = Pinecone(
    api_key=os.environ["PINECONE_API_KEY"]
)

# Path to the PDF file containing course syllabus
pdf_path = r"C:\Users\Administrator\OneDrive\Documents\GitHub\Courses-Chatbot\Revised Syllabus\Mechanical.pdf"

# Read the PDF file
pdf_reader = PdfReader(pdf_path)

# Extract text from all pages of the PDF
raw_text = ""
for i,page in enumerate(pdf_reader.pages):
    content = page.extract_text()
    if content:
        raw_text += content     # Append extracted text

# Create a Document object with extracted text
docs = [Document(page_content=raw_text)]

# Split the extracted text into smaller chunks for efficient retrieval
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,    # Maximum size of each chunk
    chunk_overlap=150,  # Overlap between chunks to maintain context
    length_function=len,
)
texts = text_splitter.split_documents(docs)

# Define a source URL for reference
documents_with_sources = []
web_url = "https://drive.google.com/file/d/1cqzriPtgSKh1fP9tHYQvM8pvSKgSEksj/view?usp=sharing"

# Add metadata to each document chunk
for doc in texts:
    doc.metadata = {
        "source": doc.metadata.get("source", "B.Tech Mechanical Engineering Course Structure"),  # Original PDF path
        "source_url": web_url,  # Provide file URL
        "text": doc.page_content,   # Store chunk content
    }
    documents_with_sources.append(doc)

# Initialize embeddings model for vectorization
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")

# Create or get existing index
index_name = "course-database"
existing_indexes = pc.list_indexes().names()
if index_name not in existing_indexes:  # Create a new Pinecone index if not present
    pc.create_index(
        name=index_name,
        dimension=1024, # Dimension of embeddings
        metric='cosine',    # Similarity metric for retrieval
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Connect to the Pinecone index
index = pc.Index(index_name)

# Initialize Pinecone vector store with the processed documents
vectorstore = LangchainPinecone.from_documents(
    documents_with_sources,
    embedding=embeddings,
    index_name=index_name
)

print("Embeddings added to Pinecone")

# # Initialize LLM model using Groq API
# llm = ChatGroq(
#     model = "deepseek-r1-distill-llama-70b",   
#     temperature = 0.3,  # Lower temperature for more deterministic answers
#     api_key = os.environ['GROQ_API_KEY']
# )

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