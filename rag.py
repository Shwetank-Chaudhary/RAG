from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Qdrant
from langchain_openai.chat_models.azure import  AzureChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core import output_parsers
from fastapi import FastAPI
from template import template
from pydantic import BaseModel, Field
import httpx

load_dotenv()

app = FastAPI()

# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
# deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
# TEXT_EMBEDDING_VERSION = os.getenv("TEXT_EMBEDDING_VERSION")

# print(TEXT_EMBEDDING_VERSION,deployment)
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

# llm = AzureChatOpenAI(
#     azure_deployment= deployment,
#     api_key = AZURE_OPENAI_KEY,
#     api_version= TEXT_EMBEDDING_VERSION,
#     azure_endpoint= AZURE_OPENAI_ENDPOINT,
#     http_client=httpx.Client(verify=False)

# )

json_parser = output_parsers.JsonOutputParser()

class SearchRequest(BaseModel):
    search_text : str  = Field(description="Search query")
    model_config = {
        "json_schema_extra":{
            "example":{
                "search_text":"Hello"
            }
        }
    }


def get_response(docs,query):
    content = [x[0].page_content for x in docs]
    # prompt = ChatPromptTemplate.from_template(template = template)
    # chain = prompt | llm | json_parser
    # return chain.invoke({"context":content,"query":query})
    return content

@app.get("/")
def root():
    return {"THis is a":"POC"}

@app.post("/search")
def search(searchRequest: SearchRequest):
    txt = searchRequest.search_text

    qdrant_client = QdrantClient(
        url = url,
        prefer_grpc = False
    )

    db = Qdrant(
        client = qdrant_client,
        embeddings= embeddings,
        collection_name=collection_name
    )

    docs = db.similarity_search_with_score(query = txt, k=5)

    return get_response(docs,txt)