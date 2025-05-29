from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_core.runnables import chain


file_path = "/Users/ravi/Downloads/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)

all_splits = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = InMemoryVectorStore(embeddings)

ids = vector_store.add_documents(documents=all_splits)

# results = vector_store.similarity_search("How many distribution centers does Nike have in the US?")

# print(results)

# results = vector_store.similarity_search("When was Nike incorporated?")

# print(results[0])

# results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
# doc, score = results[0]
# print(f"Score: {score}\n")
# print(doc)

# @chain
# def retriever(query: str) -> List[Document]:
#     return vector_store.similarity_search(query, k=1)

retriever = vector_store.as_retriever(search_kwargs={"k": 1}, search_type="similarity")


results = retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ]
)

# embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")

# results = vector_store.similarity_search_by_vector(embedding)
print(results)
