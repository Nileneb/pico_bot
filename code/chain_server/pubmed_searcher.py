from typing import List

import redis
from Bio import Entrez
from langchain.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from pymilvus import Collection


class PubMedMeshSearcher(Runnable):
    def __init__(self, embedding_model, milvus_collection: Collection, redis_client: redis.Redis):
        self.embedding_model = embedding_model
        self.milvus_collection = milvus_collection
        self.redis_client = redis_client

        # Setup Entrez
        Entrez.email = "your_email@example.com"  # Replace with actual config
        Entrez.api_key = "your_ncbi_api_key"  # Replace with actual config

    async def invoke(self, query: str, config=None):
        """Invoke the PubMed search and embedding process."""
        # Check Redis cache first
        cached_result = self.redis_client.get(query)
        if cached_result:
            return cached_result.decode("utf-8")

        # Perform PubMed search
        pmids = self.search_pubmed(query)
        if not pmids:
            return "No results found."

        # Fetch abstracts and generate embeddings
        abstracts = self.fetch_abstracts(pmids)
        embeddings = self.generate_embeddings(abstracts)

        # Store embeddings in Milvus
        self.store_in_milvus(embeddings, pmids)

        return "\n\n".join(abstracts)

    def search_pubmed(self, query: str) -> List[str]:
        """Search for PubMed articles and return PMIDs."""
        handle = Entrez.esearch(db="pubmed", term=query, retmax=10)
        results = Entrez.read(handle)
        handle.close()
        return results.get("IdList", [])

    def fetch_abstracts(self, pmids: List[str]) -> List[str]:
        """Fetch PubMed abstracts for given PMIDs."""
        handle = Entrez.efetch(db="pubmed", id=pmids, rettype="abstract", retmode="text")
        abstracts = handle.read().split("\n\n")
        handle.close()
        return abstracts

    def generate_embeddings(self, abstracts: List[str]) -> List[List[float]]:
        """Generate embeddings for the abstracts using NVIDIA embeddings."""
        embeddings = []
        for abstract in abstracts:
            embeddings.append(self.embedding_model(abstract))
        return embeddings

    def store_in_milvus(self, embeddings: List[List[float]], ids: List[str]):
        """Store embeddings in Milvus."""
        entities = [
            {"name": "embedding", "values": embeddings, "type": "float_vector"},
            {"name": "id", "values": ids, "type": "varchar"},
        ]
        self.milvus_collection.insert(entities)
