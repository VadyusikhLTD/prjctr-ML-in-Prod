from pathlib import Path
from datatypes import TableInfo
from typing import List

from config import DATA_PATH, DATA2_PATH


def get_tables_from_folder(tables_folder: Path) -> List[TableInfo]:
    return get_tables_from_path(list(tables_folder.iterdir()))


def get_tables_from_path(table_paths: List[Path]) -> List[TableInfo]:
    return sorted(
        [TableInfo(p) for p in table_paths],
        key=lambda x: x.date
    )


if __name__ == "__main__":
    print(get_tables_from_folder(DATA_PATH))
    # print(list(DATA_PATH.iterdir()))

