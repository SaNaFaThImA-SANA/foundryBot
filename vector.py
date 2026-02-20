
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# =========================
# FILE PATHS
# =========================
STARTUP_FILE = r"C:\Users\DELL\OneDrive\Documents\LLM PROJECT 1\startup datasetLLM.csv"
INDUSTRY_FILE = r"C:\Users\DELL\OneDrive\Documents\LLM PROJECT 1\startupDataset2.csv"
CONSUMER_FILE = r"C:\Users\DELL\OneDrive\Documents\LLM PROJECT 1\startupDataset3.csv"

BATCH_SIZE = 1000

# =========================
# EMBEDDINGS
# =========================
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# =========================
# VECTOR STORES
# =========================
startup_store = Chroma(
    collection_name="startup_collection",
    persist_directory="./chroma_startup_db",
    embedding_function=embeddings
)

industry_store = Chroma(
    collection_name="industry_collection",
    persist_directory="./chroma_industry_db",
    embedding_function=embeddings
)

consumer_store = Chroma(
    collection_name="consumer_collection",
    persist_directory="./chroma_consumer_db",
    embedding_function=embeddings
)

# =========================
# SURVIVAL SCORE FUNCTION
# =========================
def calculate_survival_score(row):
    score = 0

    if pd.isna(row.get("closed_at")):
        score += 3

    if row.get("age_last_funding_year", 0) and row.get("age_last_funding_year", 0) > 5:
        score += 2

    if row.get("relationships", 0) and row.get("relationships", 0) > 20:
        score += 2

    if row.get("age_last_milestone_year", 0) and row.get("age_last_milestone_year", 0) > 3:
        score += 1

    return score

# =========================
# INGEST STARTUP DATA
# =========================
def ingest_startup():
    df = pd.read_csv(STARTUP_FILE)

    documents = []
    ids = []

    for i, row in df.iterrows():

        survival_score = calculate_survival_score(row)

        content = (
            f"Startup: {row.get('name', 'N/A')} | "
            f"City: {row.get('city', 'N/A')} | "
            f"Founded: {row.get('founded_at', 'N/A')} | "
            f"Closed: {row.get('closed_at', 'Active')} | "
            f"Funding Duration: {row.get('age_last_funding_year', 'N/A')} years | "
            f"Relationships: {row.get('relationships', 'N/A')} | "
            f"Survival Score: {survival_score}"
        )

        doc = Document(
            page_content=content,
            metadata={"source": "startup"},
            id=f"startup_{i}"
        )

        documents.append(doc)
        ids.append(f"startup_{i}")

    if startup_store._collection.count() == 0:
        startup_store.add_documents(documents=documents, ids=ids)
        print("Startup dataset ingested.")

# =========================
# INGEST GENERIC DATA
# =========================
def ingest_generic(file_path, store, dataset_type):
    df = pd.read_csv(file_path)

    documents = []
    ids = []

    for i, row in df.iterrows():

        content = f"{dataset_type.upper()} DATA:\n"

        for col in df.columns:
            content += f"{col}: {row[col]}\n"

        doc = Document(
            page_content=content,
            metadata={"source": dataset_type},
            id=f"{dataset_type}_{i}"
        )

        documents.append(doc)
        ids.append(f"{dataset_type}_{i}")

    if store._collection.count() == 0:
        store.add_documents(documents=documents, ids=ids)
        print(f"{dataset_type} dataset ingested.")

# =========================
# RUN INGESTION
# =========================
ingest_startup()
ingest_generic(INDUSTRY_FILE, industry_store, "industry")
ingest_generic(CONSUMER_FILE, consumer_store, "consumer")

# =========================
# RETRIEVERS
# =========================
startup_retriever = startup_store.as_retriever(search_kwargs={"k": 5})
industry_retriever = industry_store.as_retriever(search_kwargs={"k": 3})
consumer_retriever = consumer_store.as_retriever(search_kwargs={"k": 3})
