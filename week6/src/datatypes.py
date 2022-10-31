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
    distribution_space: np.array
    distribution_pdf: np.array
    info: dict = None


@dataclass
class TableDistribution(Distribution):
    table_info: TableInfo = None


@dataclass
class Bound:
    lower: float
    upper: float


@dataclass
class Bound2D:
    x: Bound
    y: Bound


@dataclass
class Distribution2D:
    distribution_x_space: np.array
    distribution_y_space: np.array
    distribution_pdf: np.ndarray
    bounds: Bound2D
    info: dict = None


@dataclass
class TableDistribution2D(Distribution2D):
    table_info: TableInfo = None
