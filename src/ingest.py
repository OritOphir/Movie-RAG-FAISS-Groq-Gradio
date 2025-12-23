"""
ingest.py

Loads the movie dataset and converts each row into LangChain Documents
with structured metadata.
"""

import requests
import pandas as pd
from langchain_core.documents import Document


DATA_URL = "https://drive.google.com/uc?id=1n80ZoB0dUcfRM6-Dv8wS2ZvAPqbVtJxs&export=download"


def safe_str(x):
    return "" if pd.isna(x) else str(x)


def safe_float(x):
    if pd.isna(x):
        return None
    try:
        return float(str(x).replace("$", "").replace(",", ""))
    except ValueError:
        return None


def load_documents() -> list[Document]:
    """Download CSV and convert rows to LangChain Documents."""
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()

    with open("movies.csv", "wb") as f:
        f.write(response.content)

    df = pd.read_csv("movies.csv")

    documents = []
    for _, row in df.iterrows():
        content = f"""
Movie: {safe_str(row.get('Movie'))}
Year: {safe_str(row.get('Year'))}
Director: {safe_str(row.get('Director'))}
Genre: {safe_str(row.get('Imdb_genre'))}
IMDB Rating: {safe_str(row.get('IMDB Rating'))}
Metascore: {safe_str(row.get('metascore'))}
Box Office: {safe_str(row.get('Box Office Collection'))}
Cast: {safe_str(row.get('Cast'))}

Consensus:
{safe_str(row.get('Consensus'))}
""".strip()

        metadata = {
            "movie": safe_str(row.get("Movie")),
            "year": row.get("Year"),
            "genre": safe_str(row.get("Imdb_genre")),
            "director": safe_str(row.get("Director")),
            "rating": safe_float(row.get("IMDB Rating")),
        }

        documents.append(Document(page_content=content, metadata=metadata))

    return documents
