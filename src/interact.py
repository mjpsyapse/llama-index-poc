from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from urllib.parse import urlparse
import os

import nest_asyncio

nest_asyncio.apply()

OLLAMA_URL = os.environ.get("OLLAMA_URL", "https://ollama-sqa.syapse.com")
DB_URL = os.environ.get("DB_URL", "postgresql://postgres:password@db:5432/postgres")

Settings.llm = Ollama(model="qwen2", request_timeout=60.0, url=OLLAMA_URL)
Settings.embed_model = OllamaEmbedding(
    model_name="all-minilm",
    base_url=OLLAMA_URL,
    ollama_additional_kwargs={"mirostat": 0},
)


url = urlparse(DB_URL)
vector_store = PGVectorStore.from_params(
    database=url.path.replace("/", ""),
    host=url.hostname,
    user=url.username,
    password=url.password,
    port=url.port,
    table_name="rags",
    embed_dim=384,
)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
query_engine = index.as_query_engine(verbose=True,streaming=True)


response = query_engine.query(
    "How are medications defined in the client database?",

)

response.print_response_stream()



