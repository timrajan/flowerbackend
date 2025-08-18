import os
from typing import List, Optional

from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader,TextLoader,CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

from pinecone import Pinecone
import ocrmypdf
load_dotenv()
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import uuid
import time


def convert_docs():
    input_pdf = "/Users/timothyrajanalex/PycharmProjects/PythonProject/flowerbackend/pdfqaraggraph/flowers.pdf"
    output_pdf = "/Users/timothyrajanalex/PycharmProjects/PythonProject/flowerbackend/pdfqaraggraph/Cars_OCR.pdf"

    try:
        ocrmypdf.ocr(input_pdf, output_pdf, deskew=True, force_ocr=True)
        print(f"OCR complete! Text-searchable PDF saved as: {output_pdf}")
    except Exception as e:
        print(f"OCR failed: {e}")

def load_document(file_path: str):
    """
    Load a document based on its file extension.
    Args:
        file_path: Path to the document file
    Returns:
        List of Document objects
    """
    logger.info(f"Loading document: {file_path}")
    file_extension = file_path.lower().split('.')[-1]

    try:
        if file_extension == 'txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_extension == 'pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == 'csv':
            loader = CSVLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} document(s)")
        return documents

    except Exception as e:
        logger.error(f"Error loading document: {str(e)}")
        raise

def split_documents(documents):
    """
    Split documents into smaller chunks for better embedding performance.
    Args:
        documents: List of Document objects
    Returns:
        List of split Document objects
    """
    logger.info("Processing and splitting documents...")

    try:
        split_docs = split_the_documents(documents)
        logger.info(f"Split into {len(split_docs)} chunks")
        return split_docs

    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise


# Initialize text splitter
def split_the_documents(
    documents: List[Document],
    chunk_size: int = 1248,
    chunk_overlap: int = 187,
    separators: Optional[List[str]] = None,
    length_function: callable = len,
    is_separator_regex: bool = False ) -> List[Document]:

    """
    Split a list of documents into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        documents (List[Document]): List of LangChain Document objects to split
        chunk_size (int): Maximum size of each chunk in characters
        chunk_overlap (int): Number of characters to overlap between chunks
        separators (List[str], optional): List of separators to use for splitting
        length_function (callable): Function to calculate chunk length
        is_separator_regex (bool): Whether separators should be interpreted as regex

    Returns:
        List[Document]: List of split document chunks
    """

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap = chunk_overlap,
    separators = separators,
    length_function = length_function,
    is_separator_regex = is_separator_regex
    )
    # Split the documents
    split_docs = text_splitter.split_documents(documents)

    return split_docs



def upload_document_pipeline(file_path: str,
                             index_name: str,
                             namespace: Optional[str] = None):
    """
    Complete pipeline to upload a document to Pinecone with LangSmith tracing.

    Args:
        file_path: Path to the document file
        index_name: Name of the Pinecone index
        namespace: Optional namespace for organizing vectors

    Returns:
        Pinecone vector store instance
    """
    try:
        # Step 1: Load document
        documents = load_document(file_path)

        # Step 2: Process and split documents
        after_split_documents = split_documents(documents)

        # Step 3: Upload to Pinecone
        vectorstore = upload_docs_pinecone(
            documents=after_split_documents,
            index_name=index_name,
            namespace=namespace
        )

        logger.info("Document upload pipeline completed successfully")
        return vectorstore

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


def append_to_pinecone_with_unique_ids(documents, embeddings, index_name, namespace=None):
    """Append documents with guaranteed unique IDs."""

    from pinecone import Pinecone
    pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pinecone.Index(index_name)

    # Prepare data
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    embeddings_list = embeddings.embed_documents(texts)

    # Generate unique IDs using timestamp + UUID
    timestamp = int(time.time())
    vectors = []

    for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings_list, metadatas)):
        unique_id = f"{timestamp}_{uuid.uuid4().hex[:8]}_{i}"
        vectors.append({
            "id": unique_id,
            "values": embedding,
            "metadata": {
                **metadata,
                "text": text[:1000],
                "upload_timestamp": timestamp,
                "doc_index": i
            }
        })

    # Upload in batches (this will APPEND, not overwrite)
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        if namespace:
            index.upsert(vectors=batch, namespace=namespace)
        else:
            index.upsert(vectors=batch)
        print(f"✅ Uploaded batch {i // batch_size + 1}")

    print(f"✅ Appended {len(documents)} new documents to {index_name}")
    return index

embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

def upload_docs_pinecone(documents,
                       index_name: str,
                       namespace: Optional[str] = None):
    """
    Upload processed documents to Pinecone vector database.

    Args:
        documents: List of processed Document objects
        index_name: Name of the Pinecone index
        namespace: Optional namespace for organizing vectors

    Returns:
        Pinecone vector store instance
    """
    logger.info(f"Uploading {len(documents)} documents to Pinecone index: {index_name}")

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    try:
        # Initialize Pinecone client

        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(index_name)
        # Create vector store and add documents


        # Prepare data
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        embeddings_list = embeddings.embed_documents(texts)

        # Generate unique IDs using timestamp + UUID
        timestamp = int(time.time())
        vectors = []

        for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings_list, metadatas)):
            unique_id = f"{timestamp}_{uuid.uuid4().hex[:8]}_{i}"
            vectors.append({
                "id": unique_id,
                "values": embedding,
                "metadata": {
                    **metadata,
                    "text": text[:1000],
                    "upload_timestamp": timestamp,
                    "doc_index": i
                }
            })

        # Upload in batches (this will APPEND, not overwrite)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            if namespace:
                index.upsert(vectors=batch, namespace=namespace)
            else:
                index.upsert(vectors=batch)
            print(f"✅ Uploaded batch {i // batch_size + 1}")

        print(f"✅ Appended {len(documents)} new documents to {index_name}")

    except Exception as e:
        logger.error(f"Error uploading to Pinecone: {str(e)}")
        raise



def upload_docs_chroma(docpath):
    print("This is just a placeholder. We are using pinecone")


pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Get the index
index = pc.Index("flower")

# Create vectorstore (equivalent to your Chroma setup)
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace="__default__",
)
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}  # Return top 4 results
)




if __name__ == "__main__":
    #convert_docs()
    upload_document_pipeline("/Users/timothyrajanalex/PycharmProjects/PythonProject/flowerbackend/pdfqaraggraph/flowers_OCR.pdf","flower")





