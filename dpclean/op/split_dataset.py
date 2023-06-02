import os
from pathlib import Path
from typing import List

import dpdata
from dflow.python import OP, OPIO, Artifact, OPIOSign


class SplitDataset(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "dataset": Artifact(Path),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "systems": Artifact(List[Path]),
            }
        )

    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        systems = []
        for f in os.listdir(ip["dataset"]):
            path = os.path.join(ip["dataset"], f)
            if os.path.isdir(path):
                k = dpdata.LabeledSystem(path, fmt="deepmd/npy")
                nf = k.get_nframes()

                for i in range(nf):
                    target = "%s.%s" % (f, i)
                    k[i].to_deepmd_npy(target)
                    systems.append(Path(target))

        return OPIO({
            "systems": systems,
        })
