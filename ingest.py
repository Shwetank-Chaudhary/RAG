from qdrant_client import QdrantClient
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Qdrant
from qdrant_client.models import PointStruct
load_dotenv()



# TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
# TEXT_EMBEDDING_VERSION = os.getenv("TEXT_EMBEDDING_VERSION")

# client = QdrantClient(":memory:")

loader = PyPDFLoader("data.pdf")

document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)

chunks = text_splitter.split_documents(document)


model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
url = "http://localhost:6333"

embeddings = HuggingFaceBgeEmbeddings(
    model_name = model_name,
    model_kwargs = {'device':'cpu'},
    encode_kwargs = encode_kwargs
)

collection_name = "try_db"


print(collection_name)

qdrant = Qdrant.from_documents(
    chunks,
    embeddings,
    url = url,
    prefer_grpc = False,
    collection_name = collection_name
)

print("Indexed")