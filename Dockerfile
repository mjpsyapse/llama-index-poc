FROM python:3.12


RUN pip install llama-index
RUN pip install llama-index-readers-github
RUN pip install llama-index-vector-stores-postgres
RUN pip install llama-index-llms-ollama
RUN pip install llama-index-embeddings-ollama
