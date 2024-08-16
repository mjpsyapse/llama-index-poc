from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from urllib.parse import urlparse
import os

import nest_asyncio

nest_asyncio.apply()

OLLAMA_URL = "https://ollama-sqa.syapse.com/"
Settings.llm = Ollama(model="qwen2", request_timeout=60.0, url=OLLAMA_URL)
Settings.embed_model = OllamaEmbedding(
    model_name="all-minilm",
    base_url=OLLAMA_URL,
    ollama_additional_kwargs={"mirostat": 0},
)

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
DB_URL = os.environ.get("DB_URL", "postgresql://postgres:password@db:5432/postgres")

owner = "syapse"
repo = "voyager"
branch = "master"


github_client = GithubClient(github_token=GITHUB_TOKEN, verbose=True)

documents = GithubRepositoryReader(
    github_client=github_client,
    owner=owner,
    repo=repo,
    use_parser=False,
    verbose=False,
    filter_directories=(
        ["voyager"],
        GithubRepositoryReader.FilterType.INCLUDE,
    ),
    filter_file_extensions=(
        [
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
            ".ico",
            "json",
            ".ipynb",
        ],
        GithubRepositoryReader.FilterType.EXCLUDE,
    ),
).load_data(branch=branch)


# index = VectorStoreIndex.from_documents(documents)

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

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, show_progress=True
)


