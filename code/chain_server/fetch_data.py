import os

import requests
from Bio import Entrez


class Research:
    def __init__(self, email: str = "bene.linn@yahoo.de", max_results: int = 20, save_directory: str = "./"):
        self.email = email
        self.max_results = max_results
        self.save_directory = save_directory
        Entrez.email = self.email

    def search_pubmed(self, query: str) -> dict:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=self.max_results, usehistory="y")
        search_results = Entrez.read(handle)
        handle.close()
        return search_results

    def fetch_summaries(self, article_ids: list) -> list:
        summaries_handle = Entrez.esummary(db="pubmed", id=",".join(article_ids), retmode="xml")
        summaries = Entrez.read(summaries_handle)
        summaries_handle.close()
        return summaries

    def fetch_abstract(self, article_id: str) -> str:
        try:
            abstract_handle = Entrez.efetch(db="pubmed", id=article_id, rettype="abstract", retmode="text")
            abstract = abstract_handle.read()
            abstract_handle.close()
            return abstract
        except Exception as e:
            print(f"Failed to fetch abstract for article {article_id}: {e}")
            return ""

    def get_pmc_id(self, article_id: str) -> str:
        try:
            link_handle = Entrez.elink(dbfrom="pubmed", id=article_id, linkname="pubmed_pmc")
            link_results = Entrez.read(link_handle)
            link_handle.close()
            if link_results[0]["LinkSetDb"]:
                return link_results[0]["LinkSetDb"][0]["Link"][0]["Id"]
        except Exception as e:
            print(f"Failed to fetch PMC ID for article {article_id}: {e}")
        return ""

    def download_pdf(self, pmc_id: str, file_name: str) -> None:
        pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf"
        try:
            response = requests.get(pdf_url, timeout=10)
            if response.status_code == 200:
                pdf_path = os.path.join(self.save_directory, file_name)
                with open(pdf_path, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded PDF for article PMC{pmc_id} to {pdf_path}")
            else:
                print(f"PDF for article PMC{pmc_id} is not available.")
        except requests.RequestException as e:
            print(f"Failed to download PDF for article PMC{pmc_id}: {e}")

    def run_research(self, query: str) -> None:
        search_results = self.search_pubmed(query)
        pubmed_article_ids = search_results.get("IdList", [])
        total_results = int(search_results.get("Count", 0))

        if total_results == 0:
            print("No results found for the given search query.")
            return

        print(f"Found {total_results} results.")
        print("Article IDs:")
        for i, article_id in enumerate(pubmed_article_ids):
            print(f"{i + 1}. {article_id}")

        summaries = self.fetch_summaries(pubmed_article_ids[:10])
        print("\nSummaries for the first 10 articles:")
        for i, summary in enumerate(summaries):
            title = summary.get("Title", "No title available")
            authors = summary.get("AuthorList", ["No authors available"])
            source = summary.get("Source", "No source available")
            pub_date = summary.get("PubDate", "No publication date available")
            abstract = summary.get("Abstract", "No abstract available")
            doi = summary.get("DOI", "No DOI available")
            mesh_terms = summary.get("MeshHeadingList", ["No MeSH terms available"])

            print(f"{i + 1}. Title: {title}")
            print(f"   Authors: {', '.join(authors)}")
            print(f"   Source: {source}")
            print(f"   Publication Date: {pub_date}")
            print(f"   Abstract: {abstract}")
            print(f"   DOI: {doi}")
            print(f"   MeSH Terms: {', '.join(mesh_terms)}")

        available_abstracts = []
        for article_id in pubmed_article_ids:
            abstract = self.fetch_abstract(article_id)
            pmc_id = self.get_pmc_id(article_id)
            if pmc_id and abstract:
                available_abstracts.append((pmc_id, abstract))
                if len(available_abstracts) >= 5:
                    break

        for idx, (pmc_id, _) in enumerate(available_abstracts):
            self.download_pdf(pmc_id, f"article_{idx + 1}_PMC{pmc_id}.pdf")


if __name__ == "__main__":
    email = input("Enter your email: ")
    max_results = int(input("Enter the maximum number of search results to retrieve: "))
    save_directory = input("Enter the directory to save PDFs: ")

    research = Research(email=email, max_results=max_results, save_directory=save_directory)
    search_query = input("Enter your PubMed search query: ")
    research.run_research(search_query)
