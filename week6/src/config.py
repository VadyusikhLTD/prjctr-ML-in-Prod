from pathlib import Path
import numpy as np
from datatypes import Bound, Bound2D


DATA_PATH = Path('../../../pet_project/tables')
DATA2_PATH = Path('../../../PTETA/PTETA/data/local/tables')

CHERNIVTSI_CENTER = [48.300000, 25.933333]
CHERNIVTSI_CENTER_NP = np.array(CHERNIVTSI_CENTER)

## correct bounds / expanded bounds
# 48.2198 -> 48.3708 # 48.2044 -> 48.3734
# 25.8598 -> 26.0356 # 25.8402 -> 26.0542
# 25.8402, 26.0542

CHERNIVTSI_BOUND = [[48.2044, 48.3734], [25.8402, 26.0542]]
CHERNIVTSI_BOUND_2D = Bound2D(
    x=Bound(lower=48.2044, upper=48.3734),
    y=Bound(lower=25.8402, upper=26.0542)
)
