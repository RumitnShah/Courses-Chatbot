# Import required libraries
import os
from PyPDF2 import PdfReader
from langchain.schema import Document
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

if "PINECONE_API_KEY" not in os.environ:
    raise ValueError("Please set PINECONE_API_KEY in your .env file")
if "COHERE_API_KEY" not in os.environ:
    raise ValueError("Please set COHERE_API_KEY in your .env file")

# Initialize Pinecone
pc = Pinecone(
    api_key=os.environ["PINECONE_API_KEY"]
)

# Read PDF
pdf_reader = PdfReader(r"sample.pdf")

# Extract text from PDF
raw_text = ""
for i, page in enumerate(pdf_reader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# Create document
docs = [Document(page_content=raw_text)]

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)
texts = text_splitter.split_documents(docs)

# Initialize Cohere embeddings
embeddings = CohereEmbeddings(model="large")

# Create or get existing index
index_name = "courses-database"
existing_indexes = pc.list_indexes().names()
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=4096,  # Dimension for cohere-text-embedding-large
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)

# Initialize Pinecone vectorstore
vectorstore = LangchainPinecone.from_documents(
    documents=texts,
    embedding=embeddings,
    index_name=index_name
)

print("Embeddings added to Pinecone")

# # Uncomment below to use with a query and QA chain
# # Initialize LLM with increased max_tokens
# # from langchain_community.chat_models import ChatOpenAI
# # from langchain.chains import RetrievalQA
# # llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=4000, temperature=0.3)

# # Create QA chain
# # qa = RetrievalQA.from_chain_type(
# #     llm=llm,
# #     chain_type="stuff",
# #     retriever=vectorstore.as_retriever()
# # )

# # Example query
# # query = "courses in mechanical semester 1?"
# # result = qa.invoke(query)
# # print(result)
