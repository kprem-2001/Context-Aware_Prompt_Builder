from logger_config import logging
from config import PINECONE_API_KEY
from langchain.embeddings import HuggingFaceEmbeddings
from slides import slides
from langchain_core.documents import Document



logging.info("Making the slides into document for embedding and vector store")
documents = [Document(page_content=slide) for slide in slides]

logging.info("Initiaizing the embeddings")
def download_huggingface_embeddding():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2")
    return embeddings

logging.info("Downloading Embedding")
embedding = download_huggingface_embeddding()