from chromadb import ClientAPI
from chromadb.api.models.Collection import Collection

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb
import os

load_dotenv()
_client: ClientAPI | None = None
_collection: Collection | None = None
print(f"OpenAI API Key loaded: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")


def get_chroma_client() -> ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE")
        )
    return _client


def get_chroma_collection() -> Collection:
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name="flower",
        )
    return _collection


# Load and process documents
print("Loading documents...")
urls = [
    "https://aiflowershop.com/blog/1",
    "https://aiflowershop.com/blog/2",
    "https://aiflowershop.com/blog/3",
    "https://aiflowershop.com/blog/4",
    "https://aiflowershop.com/blog/5",
    "https://aiflowershop.com/blog/6",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
print(f"Loaded {len(docs_list)} documents")

# Split documents
print("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)
print(f"Created {len(doc_splits)} chunks")

# Get ChromaDB Cloud client and collection
print("Connecting to ChromaDB Cloud...")
client = get_chroma_client()
collection = get_chroma_collection()

# Test connection
try:
    client.heartbeat()
    print("✅ Connected to ChromaDB Cloud")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    exit()

# Create Chroma vectorstore using your cloud client
print("Creating vectorstore with cloud client...")
vectorstore = Chroma(
    client=client,
    collection_name="flower",
    embedding_function=OpenAIEmbeddings()
)

# -- UPLOAD --
#Upload documents to ChromaDB Cloud
print("Uploading documents to ChromaDB Cloud...")
try:
    vectorstore.add_documents(doc_splits)

    # Verify upload
    count = collection.count()
    print(f"✅ Successfully uploaded! Total documents in collection: {count}")

except Exception as e:
    print(f"❌ Upload failed: {e}")
    exit()
# -- END OF UPLOAD --

# Create retriever using the same cloud instance
print("Creating retriever...")
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}  # Return top 4 results
)

# Test the retriever
print("Testing retriever...")
try:
    test_results = retriever.get_relevant_documents("What are LLM agents?")
    print(f"✅ Retriever test successful! Found {len(test_results)} relevant documents")

    # Show first result
    if test_results:
        print(f"Sample result: {test_results[0].page_content[:200]}...")

except Exception as e:
    print(f"❌ Retriever test failed: {e}")

print("Setup complete!")