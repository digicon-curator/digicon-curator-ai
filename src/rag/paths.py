import os


ORIGINAL_DATA_PATH = "data/processed/Data.csv"
CURATED_DATA_PATH = "data/processed/Data_curated.csv"

ORIGINAL_EMBEDDING_PATH = "data/processed/embeddings.npy"
CURATED_EMBEDDING_PATH = "data/processed/embeddings_curated.npy"

ORIGINAL_INDEX_PATH = "data/processed/data.index"
CURATED_INDEX_PATH = "data/processed/data_curated.index"


def get_data_path():
    return os.getenv(
        "CURATOR_DATA_PATH",
        CURATED_DATA_PATH if os.path.exists(CURATED_DATA_PATH) else ORIGINAL_DATA_PATH,
    )


def get_embedding_path():
    return os.getenv(
        "CURATOR_EMBEDDING_PATH",
        CURATED_EMBEDDING_PATH
        if os.path.exists(CURATED_EMBEDDING_PATH)
        else ORIGINAL_EMBEDDING_PATH,
    )


def get_index_path():
    return os.getenv(
        "CURATOR_INDEX_PATH",
        CURATED_INDEX_PATH if os.path.exists(CURATED_INDEX_PATH) else ORIGINAL_INDEX_PATH,
    )
