from datetime import datetime
from dataclasses import dataclass
import numpy as np
from pathlib import Path


@dataclass
class TableInfo:
    def __init__(self, path: Path):
        self.path = path
        self.date = datetime.strptime(path.name[9:-4], '%d_%b_%Y')
        self.weekday = self.date.strftime("%A")

    path: Path
    date: datetime
    weekday: str


@dataclass
class Distribution:
    distribution_pdf: np.array
    distribution_space: np.array
    info: dict = None


@dataclass
class TableDistribution(Distribution):
    table_info: TableInfo = None
