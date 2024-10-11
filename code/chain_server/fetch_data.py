import os
import re
from typing import List, Union

import requests
import yaml
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings


class RAGChain:
    def __init__(self, config_path: str = "./config.yaml"):
        self.config = self.load_config(config_path)
        self.embedding_path = self.config.get("embedding_path", "./data/nv_embedding")
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 0)
        self.embedding_model_name = self.config.get("embedding_model", {}).get("name", "NV-Embed-QA")
        self.embedding_model_url = self.config.get("embedding_model", {}).get("url")
        if not self.embedding_model_url:
            raise ValueError("The embedding model URL must be provided in the config file.")
        self.save_directory = self.config.get("save_directory", "./pdfs")
        self.nvidia_api_key = self.get_env_variable("NVIDIA_API_KEY")

    @staticmethod
    def get_env_variable(var_name: str) -> str:
        value = os.getenv(var_name)
        if not value:
            raise EnvironmentError(f"Environment variable {var_name} is not set.")
        return value

    @staticmethod
    def load_config(config_path: str) -> dict:
        with open(config_path, "r") as config_file:
            return yaml.safe_load(config_file)

    @staticmethod
    def html_document_loader(url: Union[str, bytes]) -> str:
        try:
            response = requests.get(url)
            response.raise_for_status()
            html_content = response.text
        except requests.RequestException as e:
            print(f"Failed to load {url} due to exception {e}")
            return ""

        try:
            soup = BeautifulSoup(html_content, "html.parser")
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text()
            text = re.sub("\s+", " ", text).strip()
            return text
        except Exception as e:
            print(f"Exception {e} while loading document")
            return ""

    def create_embeddings(self, urls: List[str]) -> None:
        print(f"Storing embeddings to {self.embedding_path}")

        documents = [self.html_document_loader(url) for url in urls]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, length_function=len
        )
        texts = text_splitter.create_documents(documents)

        self.index_docs(text_splitter, texts)
        print("Generated embedding successfully")

    def index_docs(self, splitter, documents: List[str]) -> None:
        embeddings = NVIDIAEmbeddings(
            model=self.embedding_model_name,
            base_url=self.embedding_model_url,
            api_key=self.nvidia_api_key,
            truncate="END",
        )

        for document in documents:
            texts = splitter.split_text(document)
            metadatas = [{} for _ in texts]

            if os.path.exists(self.embedding_path):
                update = FAISS.load_local(
                    folder_path=self.embedding_path, embeddings=embeddings, allow_dangerous_deserialization=True
                )
                update.add_texts(texts, metadatas=metadatas)
                update.save_local(folder_path=self.embedding_path)
            else:
                docsearch = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
                docsearch.save_local(folder_path=self.embedding_path)

    @staticmethod
    def send_pubmed_search_query(query: str, retmax: int = 200) -> dict:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": retmax}
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            result = response.json().get("esearchresult", {})
            return {"idlist": result.get("idlist", []), "count": int(result.get("count", 0))}
        else:
            print(f"Failed to fetch search results: {response.status_code}")
            return {"idlist": [], "count": 0}

    def download_pdf(self, article_id: str) -> None:
        pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{article_id}/pdf"
        response = requests.get(pdf_url)

        if response.status_code == 200:
            if not os.path.exists(self.save_directory):
                os.makedirs(self.save_directory)

            pdf_path = os.path.join(self.save_directory, f"{article_id}.pdf")
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            print(f"PDF of article {article_id} downloaded successfully: {pdf_path}")
        else:
            print(f"Failed to download PDF for article {article_id}: {response.status_code}")

    def run_pico_search(self, search_query: str) -> None:
        search_results = self.send_pubmed_search_query(search_query)
        pubmed_article_ids = search_results["idlist"]
        total_results = search_results["count"]

        if total_results < 20:
            # Create embeddings and download PDFs from PubMed
            self.create_embeddings(
                [f"https://www.ncbi.nlm.nih.gov/pmc/articles/{article_id}/" for article_id in pubmed_article_ids]
            )
            for article_id in pubmed_article_ids:
                self.download_pdf(article_id)
        else:
            # Only create embeddings from the article titles
            print(f"Found {total_results} results. Downloading only titles for embedding creation.")
            self.create_embeddings(
                [f"https://www.ncbi.nlm.nih.gov/pmc/articles/{article_id}/" for article_id in pubmed_article_ids]
            )


if __name__ == "__main__":
    rag_chain = RAGChain()
    search_query = input("Enter a PubMed search query (use PICO format if applicable): ")
    rag_chain.run_pico_search(search_query)
