from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

documents = SimpleDirectoryReader("data").load_data()

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# ollama
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)

index = VectorStoreIndex.from_documents(
    documents
)

query_engine = index.as_query_engine()
response = query_engine.query("Você é um atendente do suporte técnico da Netcred. \
                              E pode ajudar apenas seguintes opções: \
                              1. Listar empresas \
                              2. Mostrar agenda da empresa \
                              Ofereça as opções ao usuário e aguarde a resposta.")
print(response)