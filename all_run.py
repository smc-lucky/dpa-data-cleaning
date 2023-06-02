import time
import getpass
from dflow import config, s3_config
from dflow.plugins import bohrium
from dflow.plugins.bohrium import TiefblueClient
import os

import json
from typing import List, Optional
from dflow import (
    Workflow,
    Step,
    argo_range,
    SlurmRemoteExecutor,
    upload_artifact,
    download_artifact,
    InputArtifact,
    OutputArtifact,
    ShellOPTemplate
)

from dflow.python import (
    PythonOPTemplate,
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    Slices
)

import subprocess, os, shutil, glob
from pathlib import Path
from typing import List

import numpy as np
import json
import shutil
import dpdata
#from dpdispatcher import Machine, Resources, Task, Submission
#from deepmd.infer import DeepPot


config["host"] = "https://workflows.deepmodeling.com"
config["k8s_api_server"] = "https://workflows.deepmodeling.com"
#bohrium.config["username"] = getpass.getpass("Bohrium username: ")
#bohrium.config["password"] = getpass.getpass("Bohrium password: ")
#bohrium.config["project_id"] = getpass.getpass("Bohrium project_id: ")

bohrium.config["username"] = "shimengchao@dp.tech"
bohrium.config["password"] = "smc124689.."
bohrium.config["project_id"] = "11176"

s3_config["repo_key"] = "oss-bohrium"
s3_config["storage_client"] = TiefblueClient()

from dflow import (
    ShellOPTemplate,
    InputParameter,
    OutputParameter,
    InputArtifact,
    OutputArtifact,
    Workflow,
    Step
)

from dflow.plugins.dispatcher import DispatcherExecutor


class RunDP(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "command": str,
            "folder": Artifact(Path),
            "add_systems": Artifact(Path)
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
         #   "command": str,
            "output_dir": Artifact(Path),
         #   "add_systems": Artifact(Path)
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        cwd = os.getcwd()
        os.chdir(op_in["folder"])
        #os.system("dp train input.json -l tmp_log && dp freeze -o graph.pb")
        os.system(op_in["command"])
        os.chdir(cwd)
        return OPIO({
            "output_dir": Path(op_in["folder"])
         #   "model": Path("./graph.pb"),
        })

def submit_dp(  name = "run-dp-train", str_add="add_data", str_m0="m0-train-dpa1", scass_type = "c8_m32_1 * NVIDIA V100",
                platform = "ali", com="dp train input.json -l tmp_log && dp freeze -o graph.pb",
                image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6"  ):

    dispatcher_executor = DispatcherExecutor(
            machine_dict={
                "batch_type": "Bohrium",
                "context_type": "Bohrium",
                "remote_profile": {
                    "input_data": {
                        "job_type": "container",
                        "platform": platform,
                        "scass_type": scass_type
                    },
                },
            },
        )

    wf = Workflow(name="steps")
    train_step = Step(
            name,
            PythonOPTemplate(RunDP, image),
            parameters={"command": com},
            artifacts={"folder": upload_artifact([str_m0]),
                       "add_systems": upload_artifact([str_add])
                       },
            executor=dispatcher_executor,
        )

    wf.add(train_step)
    wf.submit()
    while wf.query_status() in ["Pending", "Running"]:
        time.sleep(1)

    assert(wf.query_status() == "Succeeded")
    step = wf.query_step(name)[0]
    assert(step.phase == "Succeeded")

    download_artifact(step.outputs.artifacts["output_dir"])


def div(path, old_path, new_path):
    os.chdir(path)
    all_folder = []

    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            all_folder.append(item)

    for ff in all_folder:
        k = dpdata.LabeledSystem(ff, fmt="deepmd/npy")
        nf = k.get_nframes()

        for i in range(nf):
            ii = new_path + "/" + ff + "." + str(i)
            k[i].to_deepmd_npy_mixed(ii)
            #print(i, ii)
    os.chdir(old_path)

def sort_and_output_indices(input_list):
    sorted_list = sorted(enumerate(input_list), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_list]


def screen_samples(dp, idx, new_path, train_path, graph_pb, n, threshold):
    rmse_f = []
    samples_idx = []
    add_samples = []
    #dp = DeepPot(pb_model)

    if len(os.listdir(new_path)) > 0:

        ### s2-dptest ####
        ff = "out-dptest" + str(idx)
        str1 =  dp + " test -m "  + graph_pb  + " -s " + "../../add_systems/" + new_path + " -l " + ff 
        print()
        print(str1) 
        print()
        #os.system(str_1)
        submit_dp( name = "run-dp-test", str_add= new_path, str_m0=train_path, scass_type = "c8_m32_1 * NVIDIA V100",
                   platform = "ali", com=str1, image="registry.dp.tech/dptech/deepmd-kit:2.2.1-cuda11.6")

        os.system(f"mv {train_path}/{ff} .")
        print("dp test finished!")
        #os.system("sed -i '/DEEPMD/d' " + ff)
        fp = open(ff, 'r')
        for ii in fp:
            if "testing system" in ii:
                item = ii.split()[-1].split("/")[-1]
                samples_idx.append(item)

            if "Force  RMSE" in ii:
                err_f = float(ii.split()[-2])
                rmse_f.append(err_f)

        #print(rmse_f)
        ### choose max n else:
        f_max = max(rmse_f[:-1])
        f_ave = np.mean(rmse_f[:-1])
        f_min = min(rmse_f[:-1])

        print()
        print('max force (eV/A): ', f_max)
        print('ave force (eV/A): ', f_ave)
        print('min force (eV/A): ', f_min)

        print()
        sorted_indices = sort_and_output_indices(rmse_f[:-1])
        #print(sorted_indices)
        if f_max - f_ave > threshold * f_ave:
            ### choose max force: 50

            if len(sorted_indices) < n:
                for i in range(len(sorted_indices)):
                    add_samples.append(samples_idx[sorted_indices[i]])
            else:
                for i in range(n):
                    add_samples.append(samples_idx[sorted_indices[i]])
            #print(add_samples)
        else:
            print("Threshold value is arrived, Do not add new samples, Job Done!")
            return []
        return add_samples

    else:
        print("All samples were added")
        return []

def do_input(element, add_samples, add_folder, path, \
             se_type, sel, rcut, rcut_smth, \
             spe, lpe, spf, lpf, spv, lpv, \
             train_steps, beg_lr, decay, end_lr, \
             input_json="input.json", output_json="out.json" ):
    ### init-model: train steps  ###

    with open(input_json, 'r') as file:
        data = json.load(file)

    #####  set up   #############
    #data["model"]["type_map"] = element
    #data["model"]["descriptor"]["type"] = se_type
    #data["model"]["descriptor"]["sel"] = sel
    #data["model"]["descriptor"]["rcut_smth"] = rcut_smth
    #data["model"]["descriptor"]["rcut"] = rcut

    #print(data["loss"])
    #data["loss"]["start_pref_e"] = spe
    #data["loss"]["limit_pref_e"] = lpe
    #data["loss"]["start_pref_f"] = spf
    #data["loss"]["limit_pref_f"] = lpf
    #data["loss"]["start_pref_v"] = spv
    #data["loss"]["limit_pref_v"] = lpv

    ### learning_rate
    #data["learning_rate"]["start_lr"] = beg_lr
    #data["learning_rate"]["decay_steps"] = decay
    #data["learning_rate"]["stop_lr"] = end_lr

    s1 = data["training"]["training_data"]["systems"]
    arr = [0] * len(s1)

    n = len(add_samples)
    brr = [1 / n] * n
    arr.extend(brr)
    for i in range(len(add_samples)):
        s1.append("../../add_systems/" + add_folder + "/" + add_samples[i])

    #####data["training"]["training_data"]["sys_probs"] = arr
    data["training"]["training_data"]["systems"] = s1

    #data["training"]["numb_steps"] = train_steps
    #data["training"]["training_data"]["batch_size"] = "auto"

    with open(output_json, 'w') as file:
        json.dump(data, file, indent=2)


def prepare_file(path1, path2, train_folder, pre_add_folder, add_folder, add_samples ):
    shutil.copytree(path1, train_folder)
    #  /root/work/iter.000000/train /root/work /root/work/iter.000000/train
    #  /root/work/a-ternary/one_frame_set add_data
    # rm old input.json
    #/root/workSpace/AlMgCu/smc-simplify/iter-screen/iter.000000 train
    #/root/workSpace/AlMgCu/smc-simplify/iter-screen/m0-train-dpa1 path1
    str1 = "mv " + input_json + " " + train_folder
    os.system(str1)

    #str2 = "mkdir -p " + path2 + "/" + add_folder
    #os.system(str2)

    for i in add_samples:
        str3 = 'mv ' + pre_add_folder + "/" + i + " " + path2 + "/" + add_folder
        #print(str3)
        os.system(str3)


def run_dp_train(dp, work_folder, train_folder, add_folder, input_json, graph_pb ):

    #os.chdir(work_folder + "/" + train_folder)
    #str1 = dp + ' train --init-model model.ckpt ' + input_json + '  && ' + dp +' freeze -o ' + graph_pb
    str1 = "cd " + train_folder + " && " + dp + ' train ' + input_json + " -l tmp_log " + '  && ' + dp +' freeze -o ' + graph_pb
    print(str1)

    #os.system(str1)
    submit_dp( name = "run-dp-train", str_add=add_folder, str_m0=work_folder, scass_type = "c8_m32_1 * NVIDIA V100",
               platform = "ali", com=str1, image="registry.dp.tech/dptech/deepmd-kit:2.2.1-cuda11.6")

    #os.chdir("../..")

if __name__ == "__main__":
    cur_path = os.getcwd()
    print(cur_path)
    os.system("date")
    dp = "/opt/deepmd-kit-2.2.1/bin/dp"

    element = []
    se_type = "se_atten"
    sel = 120
    rcut_smth = 0.5
    rcut = 6.0

    #data["loss"] ########
    spe = 0.02
    lpe = 1
    spf = 1 
    lpf = 1
    spv = 0
    lpv = 0

    ### learning_rate  ######
    beg_lr = 1e-3
    decay = 5000
    end_lr = 3.51e-08
    steps = 1000000

    number_samples = 5   #19391
    threshold = 0.0000000020
    input_json = "input.json"
    graph_pb = "graph.pb"

    path_train = []
    path_train.append("m0-train-dpa1")

    path_train_set = "pre"
    new_path = "rest_train"

    add_folder = "add_data"
    str2 = "mkdir -p " + add_folder
    os.system(str2)
    ##################  restart ######################
    i = 0
    j = 0    ### trained? j = 0, no;  else yes
    #################################################

    model_path = path_train[0]
    train_folder = "./"
    work_folder = path_train[0]
    # div(cur_path + "/" + path_train_set, cur_path, new_path)
    # random data for training
    str1 = "cd " + path_train[0] + ";" + dp + " train input.json -l tmp_log && dp freeze -o graph.pb && cd .. && pwd"
    print(path_train[-1], type(path_train[-1]))

    submit_dp( name = "run-dp-train", str_add=add_folder, str_m0=path_train[-1], scass_type = "c8_m32_1 * NVIDIA V100",
               platform = "ali", com=str1, image="registry.dp.tech/dptech/deepmd-kit:2.2.1-cuda11.6")

    #os.system("cp " + path_train[-1] + "/" + graph_pb + " " + graph_pb)

    flag = 0
    pb_model = []
    str_iter = "iter."
    while flag == 0:
        if j == 0:
            pb_model.append(model_path + "/" + graph_pb)
        else:
            pb_model.append(str_iter + str(f"%06d" % (i-1)) + "/" + train_folder + "/" + graph_pb)
            path_train.append(  str_iter + str(f"%06d" % (i-1))  + "/" + train_folder)
            j = 0

        #add_samples = screen_samples(dp, i, cur_path + "/" + path_train_set + "/" + new_path, \
        #                             pb_model[-1], number_samples, threshold)

        add_samples = screen_samples(dp, i, path_train_set + "/" + new_path,  work_folder, \
                                     graph_pb, number_samples, threshold)

        print("iter: ", i, ", add samples number: ", len(add_samples))

        if len(add_samples) != 0:
            do_input(element, add_samples, add_folder, path_train_set + "/" + new_path,
                     se_type, sel, rcut, rcut_smth, \
                     spe, lpe, spf, lpf, spv, lpv, \
                     steps, beg_lr, decay, end_lr, \
                     input_json=cur_path + "/" + path_train[-1] + "/" + input_json, \
                     output_json=cur_path + "/" + input_json)

            work_folder = str_iter + str(f"%06d" % i)
            model_path = work_folder + "/" + train_folder

            prepare_file( cur_path + "/" + path_train[-1], cur_path, \
                          cur_path + "/" + work_folder + "/" + train_folder, \
                          cur_path + "/" + path_train_set + "/" + new_path, add_folder, add_samples)
            run_dp_train(dp, work_folder, train_folder, add_folder, input_json, graph_pb )

            path_train.append(work_folder + "/" + train_folder)

            #os.system("cp " + path_train[-1] + "/" + graph_pb + " " + graph_pb)
            i = i + 1
        else:
            flag = 1
            print("Job Done!")
