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


@dataclass
class TableDistribution(Distribution):
    # def __init__(self, table_info: TableInfo, distribution_pdf: np.array, distribution_space: np.array):
    #     self.table_info = table_info
    #     self.distribution_pdf = distribution_pdf
    #     self.distribution_space = distribution_space

    table_info: TableInfo



@dataclass
class ReferenceDistribution(Distribution):
    info: dict = None


