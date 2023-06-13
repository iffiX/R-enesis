import os

DATA_DIR = os.path.join(
    os.path.realpath(os.path.dirname(__file__)),
    os.path.pardir,
    os.path.pardir,
    "data",
)


def get_cache_directory(cache_name: str):
    dir = os.path.join(DATA_DIR, "cache", cache_name)
    os.makedirs(dir, exist_ok=True)
    return dir


def get_data_directory(data_name: str):
    dir = os.path.join(DATA_DIR, data_name)
    os.makedirs(dir, exist_ok=True)
    return dir
