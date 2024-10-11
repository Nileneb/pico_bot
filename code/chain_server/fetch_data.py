import os
import re
from typing import List, Union
from urllib.parse import (
    quote,  # Dieser Import ist entfernt worden, da er nicht verwendet wurde
)

import requests
import yaml
from Bio import Entrez
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
        self.embedding_model_name = self.config["embedding_model"]["name"]
        self.embedding_model_url = self.config["embedding_model"]["url"]
        self.save_directory = self.config.get("save_directory", "./pdfs")
        self.nvidia_api_key = self.get_env_variable("NVIDIA_API_KEY")
        Entrez.email = self.config.get("email", "bene.linn@yahoo.de")

    @staticmethod
    def get_env_variable(var_name: str) -> str:
        value = os.getenv(var_name)
        if not value:
            raise EnvironmentError(f"Environment variable {var_name} is not set.")
        return value

    @staticmethod
    def load_config(config_path: str) -> dict:
        with open(config_path, "r", encoding="utf-8") as config_file:  # Encoding hinzugefügt
            return yaml.safe_load(config_file)

    @staticmethod
    def html_document_loader(url: Union[str, bytes]) -> str:
        try:
            response = requests.get(url, timeout=10)  # Timeout hinzugefügt, um unendliches Warten zu vermeiden
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
            text = re.sub(r"\s+", " ", text).strip()
            return text
        except Exception as e:
            print(f"Exception {e} while loading document")
            return ""

    def create_embeddings(self, texts: List[str]) -> None:
        print(f"Storing embeddings to {self.embedding_path}")

        embeddings = NVIDIAEmbeddings(
            model=self.embedding_model_name,
            base_url=self.embedding_model_url,
            api_key=self.nvidia_api_key,
            truncate="END",
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, length_function=len
        )

        for text in texts:
            chunks = text_splitter.split_text(text)
            if not chunks:
                continue
            metadatas: List[dict] = [{} for _ in chunks]  # Typannotation hinzugefügt

            if os.path.exists(self.embedding_path):
                update = FAISS.load_local(
                    folder_path=self.embedding_path, embeddings=embeddings, allow_dangerous_deserialization=True
                )
                update.add_texts(chunks, metadatas=metadatas)
                update.save_local(folder_path=self.embedding_path)
            else:
                docsearch = FAISS.from_texts(chunks, embedding=embeddings, metadatas=metadatas)
                docsearch.save_local(folder_path=self.embedding_path)

    @staticmethod
    def send_pubmed_search_query(query: str, retmax: int = 200, usehistory: bool = True) -> dict:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": retmax,
            "usehistory": "y" if usehistory else "n",
        }
        response = requests.get(base_url, params=params, timeout=10)  # Timeout hinzugefügt
        if response.status_code == 200:
            result = response.json().get("esearchresult", {})
            return {
                "idlist": result.get("idlist", []),
                "count": int(result.get("count", 0)),
                "webenv": result.get("webenv"),
                "query_key": result.get("querykey"),
            }
        else:
            print(f"Failed to fetch search results: {response.status_code}")
            return {"idlist": [], "count": 0}

    @staticmethod
    def download_summary(webenv: str, query_key: str) -> List[str]:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params = {"db": "pubmed", "query_key": query_key, "WebEnv": webenv, "retmode": "json"}
        response = requests.get(base_url, params=params, timeout=10)  # Timeout hinzugefügt
        if response.status_code == 200:
            doc_summaries = response.json().get("result", {})
            return [doc_summaries[doc_id]["title"] for doc_id in doc_summaries if doc_id != "uids"]
        else:
            print(f"Failed to download summaries: {response.status_code}")
            return []

    def convert_pubmed_to_pmc(self, pubmed_ids: List[str]) -> List[str]:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
        pmc_ids = []

        params = {"dbfrom": "pubmed", "db": "pmc", "id": ",".join(pubmed_ids), "retmode": "json"}
        response = requests.get(base_url, params=params, timeout=10)  # Timeout hinzugefügt

        if response.status_code == 200:
            linksets = response.json().get("linksets", [])
            for linkset in linksets:
                linkset_ids = linkset.get("linksetdbs", [])
                if linkset_ids:
                    pmc_ids.extend([link["id"] for link in linkset_ids[0]["links"]])
        else:
            print(f"Failed to convert PubMed IDs to PMC IDs: {response.status_code}")
            print(f"Response content: {response.text}")

        if not pmc_ids:
            print(f"No PMC IDs found for the given PubMed IDs: {pubmed_ids}")

        return pmc_ids

    def download_pdf(self, article_id: str) -> None:
        pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{article_id}/pdf"
        response = requests.get(pdf_url, timeout=10)  # Timeout hinzugefügt

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
        webenv = search_results.get("webenv")
        query_key = search_results.get("query_key")

        if total_results == 0:
            print("No results found for the given search query.")
            return

        print(f"Found {total_results} results.")
        if not isinstance(webenv, str) or not isinstance(query_key, str):
            print("Error: WebEnv or Query Key is not valid.")
            return

        summaries = self.download_summary(webenv, query_key)
        print("Displaying the first 10 summaries:")
        for i, summary in enumerate(summaries[:10]):
            print(f"{i + 1}. {summary}")

        selected_ids = input("Enter the numbers of the articles you want to process (comma-separated): ")
        selected_indices = [int(x.strip()) - 1 for x in selected_ids.split(",") if x.strip().isdigit()]
        selected_article_ids = [pubmed_article_ids[i] for i in selected_indices if 0 <= i < len(pubmed_article_ids)]

        if not selected_article_ids:
            print("No valid articles selected.")
            return

        # Convert PubMed IDs to PMC IDs
        pmc_article_ids = self.convert_pubmed_to_pmc(selected_article_ids)

        if not pmc_article_ids:
            print("No valid PMC articles found for the selected PubMed IDs.")
            return

        urls = [f"https://www.ncbi.nlm.nih.gov/pmc/articles/{article_id}/" for article_id in pmc_article_ids]
        texts = [self.html_document_loader(url) for url in urls]

        if all(text == "" for text in texts):
            print("No valid articles to process for embeddings.")
            return

        self.create_embeddings([text for text in texts if text])

        for article_id in pmc_article_ids:
            self.download_pdf(article_id)


if __name__ == "__main__":
    rag_chain = RAGChain()
    search_query = input("Enter a PubMed search query (use PICO format if applicable): ")
    rag_chain.run_pico_search(search_query)
