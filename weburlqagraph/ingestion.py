from chromadb import ClientAPI
from chromadb.api.models.Collection import Collection

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb
import os
import asyncio
from playwright.async_api import async_playwright
from langchain.schema import Document

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


async def load_url_with_playwright(url):
    """Load a single URL with Playwright - mimics WebBaseLoader.load()"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.set_extra_http_headers({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        try:
            await page.goto(url, wait_until='networkidle')
            await page.wait_for_timeout(2000)

            text_content = await page.evaluate('() => document.body.innerText')

            doc = Document(
                page_content=text_content,
                metadata={"source": url}
            )

            await browser.close()
            return [doc]

        except Exception as e:
            print(f"Error loading {url}: {str(e)}")
            await browser.close()
            return []


async def load_all_urls(urls):
    """Load all URLs concurrently"""
    tasks = [load_url_with_playwright(url) for url in urls]
    return await asyncio.gather(*tasks)









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
    "https://aiflowershop.com/",
    "https://aiflowershop.com/contact",
    "https://aiflowershop.com/blog",
    "https://aiflowershop.com/chat",
    "https://aiflowershop.com/qa",
    "https://aiflowershop.com/bouquets",
    "https://aiflowershop.com/pots",
    "https://aiflowershop.com/fresh-flowers",
    "https://aiflowershop.com/accessories",
    "https://aiflowershop.com/specialty",
    "https://aiflowershop.com/preserved",
    "https://aiflowershop.com/blog/1",
    "https://aiflowershop.com/blog/2",
    "https://aiflowershop.com/blog/3",
    "https://aiflowershop.com/blog/4",
    "https://aiflowershop.com/blog/5",
    "https://aiflowershop.com/blog/6",
]

# docs = [WebBaseLoader(url).load() for url in urls]
# docs_list = [item for sublist in docs for item in sublist]
# print(f"Loaded {len(docs_list)} documents")

docs = asyncio.run(load_all_urls(urls))
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