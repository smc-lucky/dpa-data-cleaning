from pathlib import Path
from typing import List, Tuple

import numpy as np
from dpclean.op import SelectSamples


class DPSelectSamples(SelectSamples):
    def load_model(self, model: Path):
        from deepmd.infer import DeepPot
        self.dp = DeepPot(model)

    def evaluate(self,
                 coord: np.ndarray,
                 cell: np.ndarray,
                 atype: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        e, f, v = self.dp.eval(coord, cell, atype)
        return e, f, v
