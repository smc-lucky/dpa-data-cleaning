from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import List, Tuple

import dpdata
import numpy as np
from dflow.python import OP, OPIO, Artifact, OPIOSign


class SelectSamples(OP, ABC):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "systems": Artifact(List[Path]),
                "model": Artifact(Path),
                "max_selected": int,
                "threshold": float,
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "selected_systems": Artifact(List[Path]),
                "remaining_systems": Artifact(List[Path]),
                "n_selected": int,
                "n_remaining": int,
                "converged": bool,
            }
        )

    @abstractmethod
    def load_model(self, model: Path):
        pass

    @abstractmethod
    def evaluate(self,
                 coord: np.ndarray,
                 cell: np.ndarray,
                 atype: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        rmse_f = []
        self.load_model(ip["model"])

        if len(ip["systems"]) == 0:
            return OPIO({
                "selected_systems": [],
                "remaining_systems": [],
                "n_selected": 0,
                "n_remaining": 0,
                "converged": False,
            })

        for sys in ip["systems"]:
            k = dpdata.LabeledSystem(sys, fmt="deepmd/npy")
            cell0 = k[0].data["cells"]
            [a0, b0, c0] = cell0.shape
            cell = cell0.reshape(b0, c0).reshape([1, -1])

            coord0 = k[0].data["coords"]
            [a1, b1, c1] = coord0.shape
            coord = coord0.reshape(b1, c1).reshape([1, -1])


            force0 = k[0].data["forces"]
            [a2, b2, c2] = force0.shape

            atype = k[0].data["atom_types"]
            e, f, v = self.evaluate(coord, cell, atype)

            lx = 0
            for i in range(b2):
                lx += (force0[0][i][0] - f[0][i][0]) ** 2 + \
                        (force0[0][i][1] - f[0][i][1]) ** 2 + \
                        (force0[0][i][2] - f[0][i][2]) ** 2
            err_f = ( lx / b2 / 3 ) ** 0.5
            rmse_f.append(err_f)

        f_max = max(rmse_f)
        f_ave = np.mean(rmse_f)
        f_min = min(rmse_f)

        logging.info('max force (eV/A): ', f_max)
        logging.info('ave force (eV/A): ', f_ave)
        logging.info('min force (eV/A): ', f_min)

        if f_max - f_ave <= ip["threshold"] * f_ave:
            return OPIO({
                "selected_systems": [],
                "remaining_systems": ip["systems"],
                "n_selected": 0,
                "n_remaining": len(ip["systems"]),
                "converged": True,
            })

        sorted_indices = np.argsort(rmse_f)
        selected_systems = []
        remaining_systems = []
        for i in range(len(sorted_indices)):
            if i < ip["max_selected"]:
                selected_systems.append(ip["systems"][sorted_indices[i]])
            else:
                remaining_systems.append(ip["systems"][sorted_indices[i]])

        return OPIO({
                "selected_systems": selected_systems,
                "remaining_systems": remaining_systems,
                "n_selected": len(selected_systems),
                "n_remaining": len(remaining_systems),
                "converged": False,
            })
