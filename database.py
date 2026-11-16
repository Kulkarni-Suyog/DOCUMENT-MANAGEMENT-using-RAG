# database.py

import os

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings



embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key="AQ.Ab8RN6KHh36-EvqMEO6saNFlevby23WG_JF5cKuMJvkEf0fyFA"
)

vector_store = Chroma(
    collection_name="myCollection",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


def addData(data: list):
    vector_store.add_texts(
        texts=data
    )
    print("Data stored successfully!")


def getData():
    data = vector_store.get()
    return data


def askQuestion(askedQuery: str):
    chunks = vector_store.similarity_search(
        query=askedQuery,
        k=2
    )
    return chunks