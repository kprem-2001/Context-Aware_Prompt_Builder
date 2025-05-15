from logger_config import logging
from config import PINECONE_API_KEY
from langchain.embeddings import HuggingFaceEmbeddings
from slides import slides
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from embedding import embedding, documents


logging.info("Initializing the Chroma DB ")
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "hii"

pc.create_index(
    name=index_name,
    dimension=768,  # Replace with your model dimensions
    metric="cosine",  # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)


logging.info("Creating Vector Store")
docsearch = PineconeVectorStore.from_documents(
    documents=documents,
    index_name=index_name,
    embedding=embedding
)