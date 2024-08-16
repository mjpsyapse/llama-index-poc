from llama_index.core import VectorStoreIndex
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex

from util.common import GITHUB_TOKEN, vector_store

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


storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, show_progress=True
)


