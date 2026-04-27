from pathlib import Path
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = (_HERE / ".." / ".." / ".." / "data").resolve()


# =========================
# strict UTF-8 CSV reader
# =========================
def read_csv_utf8(path: Path, **kwargs) -> pd.DataFrame:
    """
    Strict UTF-8 CSV reader.
    Does NOT modify files.
    Raises clear error if file is not UTF-8 encoded.
    """
    try:
        return pd.read_csv(path, encoding="utf-8", **kwargs)
    except UnicodeDecodeError as e:
        raise RuntimeError(
            f"CSV is not UTF-8 encoded:\n{path}\n"
            f"Please convert it to UTF-8 (Excel: 'CSV UTF-8')."
        ) from e


# =========================
# loading functions
# =========================
def load_table_list(root: Path = _ROOT) -> pd.DataFrame:
    return read_csv_utf8(root / "tables" / "table_list.csv")


def load_table(table_id: str, root: Path = _ROOT) -> pd.DataFrame:
    return read_csv_utf8(root / "tables" / table_id)


def load_tables(root: Path = _ROOT) -> dict:
    table_list = load_table_list(root)
    return {table_id: load_table(table_id, root)
            for table_id in table_list["table_id"]}


def load_label_class(root: Path = _ROOT) -> pd.DataFrame:
    return read_csv_utf8(root / "labels" / "ground_truth_class.csv")


def load_label_cnp(root: Path = _ROOT) -> pd.DataFrame:
    return read_csv_utf8(root / "labels" / "ground_truth_class_n_property.csv")