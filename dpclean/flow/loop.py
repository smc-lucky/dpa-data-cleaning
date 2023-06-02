from importlib import import_module

from dflow import (InputArtifact, InputParameter, Step, Steps, Workflow,
                   upload_artifact)
from dflow.python import PythonOPTemplate
from dflow.plugins.dispatcher import DispatcherExecutor
from dpclean.op import SplitDataset


class ActiveLearning(Steps):
    def __init__(self, select_op, train_op, select_image, train_image,
                 select_executor=None, train_executor=None):
        super().__init__("active-learning-loop")
        self.inputs.parameters["iter"] = InputParameter(value=0, type=int)
        self.inputs.parameters["max_selected"] = InputParameter(type=int)
        self.inputs.parameters["threshold"] = InputParameter(type=float)
        self.inputs.parameters["train_params"] = InputParameter(type=dict)
        self.inputs.artifacts["candidate_systems"] = InputArtifact()
        self.inputs.artifacts["current_systems"] = InputArtifact()
        self.inputs.artifacts["valid_systems"] = InputArtifact()
        self.inputs.artifacts["model"] = InputArtifact()

        select_step = Step(
            "select-samples",
            template=PythonOPTemplate(select_op, image=select_image),
            parameters={"max_selected": self.inputs.parameters["max_selected"],
                        "threshold": self.inputs.parameters["threshold"]},
            artifacts={"systems": self.inputs.artifacts["candidate_systems"],
                       "model": self.inputs.artifacts["model"]},
            executor=select_executor,
            key="iter-%s-select" % self.inputs.parameters["iter"],
        )
        self.add(select_step)

        train_step = Step(
            "train",
            template=PythonOPTemplate(train_op, image=train_image),
            parameters={"train_params": self.inputs.parameters["train_params"]},
            artifacts={"current_systems": self.inputs.artifacts["current_systems"],
                       "added_systems": select_step.outputs.artifacts["selected_systems"],
                       "valid_systems": self.inputs.artifacts["valid_systems"],
                       "model": self.inputs.artifacts["model"]},
            when="%s > 0" % select_step.outputs.parameters["n_selected"],
            executor=train_executor,
            key="iter-%s-train" % self.inputs.parameters["iter"],
        )
        self.add(train_step)

        next_step = Step(
            "next-loop",
            template=self,
            parameters={"iter": self.inputs.parameters["iter"] + 1,
                        "max_selected": self.inputs.parameters["max_selected"],
                        "threshold": self.inputs.parameters["threshold"],
                        "train_params": self.inputs.parameters["train_params"]},
            artifacts={"candidate_systems": select_step.outputs.artifacts["remaining_systems"],
                       "current_systems": train_step.outputs.artifacts["current_systems"],
                       "valid_systems": self.inputs.artifacts["valid_systems"],
                       "model": train_step.outputs.artifacts["model"]},
            when="%s > 0" % select_step.outputs.parameters["n_selected"],
        )
        self.add(next_step)


def import_func(s : str):
    fields = s.split(".")
    if fields[0] == __name__ or fields[0] == "":
        fields[0] = ""
        mod = import_module(".".join(fields[:-1]), package=__name__)
    else:
        mod = import_module(".".join(fields[:-1]))
    return getattr(mod, fields[-1])


def build_workflow(config):
    dataset = config["dataset"]
    init_data = config.get("init_data", [])
    valid_data = config["valid_data"]
    model = config["model"]
    split = config.get("split", {})
    select = config["select"]
    train = config["train"]

    split_op = split.get("op")
    if split_op is None:
        split_op = SplitDataset
    else:
        split_op = import_func(split_op)
    split_image = split.get("image", "dptechnology/dpdata")
    split_executor = split.get("executor")
    if split_executor is not None:
        split_executor = DispatcherExecutor(**split_executor)

    select_op = import_func(select["op"])
    select_image = select["image"]
    select_executor = select.get("executor")
    if select_executor is not None:
        select_executor = DispatcherExecutor(**select_executor)
    max_selected = select["max_selected"]
    threshold = select["threshold"]

    train_op = import_func(train["op"])
    train_image = train["image"]
    train_executor = train.get("executor")
    if train_executor is not None:
        train_executor = DispatcherExecutor(**train_executor)
    train_params = train["params"]

    wf = Workflow("clean-data")
    dataset_artifact = upload_artifact(dataset)
    split_step = Step(
        "split-dataset",
        template=PythonOPTemplate(split_op, image=split_image),
        artifacts={"dataset": dataset_artifact},
        executor=split_executor,
        key="split-dataset"
    )
    wf.add(split_step)

    active_learning = ActiveLearning(select_op, train_op, select_image,
                                     train_image, select_executor,
                                     train_executor)

    model_artifact = upload_artifact(model)
    init_data_artifact = upload_artifact(init_data)
    valid_data_artifact = upload_artifact(valid_data)
    loop_step = Step(
        "active-learning-loop",
        template=active_learning,
        parameters={"max_selected": max_selected,
                    "threshold": threshold,
                    "train_params": train_params},
        artifacts={"current_systems": init_data_artifact,
                   "candidate_systems": split_step.outputs.artifacts["systems"],
                   "valid_systems": valid_data_artifact,
                   "model": model_artifact},
    )
    wf.add(loop_step)
    return wf
