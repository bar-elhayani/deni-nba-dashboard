from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)
