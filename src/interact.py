from llama_index.core import VectorStoreIndex
from util.common import vector_store

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
query_engine = index.as_query_engine(verbose=True)

response = query_engine.query(
    "What do the `process_table` and `load_dataset` functions do?",
)

print(str(response))
