import json
import os

from dflow.python import OP, OPIO
from dpclean.op import RunTrain


class RunDPTrain(RunTrain):
    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        params = ip["train_params"]
        params["training"]["training_data"]["systems"] = [
            str(s) for s in ip["current_systems"] + ip["added_systems"]]
        params["training"]["validation_data"]["systems"] = [
            str(s) for s in ip["valid_systems"]]

        with open("input.json", "w") as f:
            json.dump(params, f, indent=2)
        
        cmd = 'dp train --init-frz-model %s input.json && dp freeze -o graph.pb' % ip["model"]
        ret = os.system(cmd)
        assert ret == 0, "Command %s failed" % cmd

        return OPIO({
            "current_systems": ip["current_systems"] + ip["added_systems"],
            "model": "graph.pb",
            "output_dir": ".",
        })
