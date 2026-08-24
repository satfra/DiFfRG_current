import glob
import os
import shutil
from multiprocessing import Pool
import subprocess

from DiFfRG.utilities import globalize
from DiFfRG.utilities import is_close
import DiFfRG.file_io as io

def get_all_sims(folder) -> list:
    files_hdf5 = glob.glob(folder + "*.h5")
    sims = [io.SimulationData(f) for f in files_hdf5]
    return sims

def get_all_finished_sims(folder) -> list:
    sims = get_all_sims(folder)
    finished_sims = [s for s in sims if s.is_finished()]
    return finished_sims

def sim_exists(sim_set, param_list) -> bool:
    for sim in sim_set:
        sim_config = sim.params
        match = True
        # param_list's names are path-like, so we need to split them by "/" and traverse the config dict accordingly
        for param in param_list:
            param_name = param[0].split("/")
            param_value = param[1]
            config_value = sim_config
            for name in param_name:
                if name in config_value:
                    config_value = config_value[name]
                else:
                    match = False
                    break
            if not match:
                break
            if is_close(config_value, param_value):
                match = False
                break
        if match:
            return True
    return False

def get_command(exe, param_list, folder="", add_params="") -> str:
    if len(param_list) < 1:
        raise Exception(
            "When adding a point in the PD, you need to give as argument a list of [name, value] lists."
        )
    name = get_name(param_list)
    attach = ""
    for param in param_list:
        if len(param) != 2:
            raise Exception(
                "When adding a point in the PD, you need to give as argument a list of [name, value] lists."
            )
        attach = f"{attach} -sd {param[0]}={param[1]}"
    if folder != "":
        return f"{exe} -ss /output/name={name} {attach} -ss /output/folder={folder} {add_params}"
    return f"{exe} -o {name} {attach} {add_params}"

def run_point(exe, param_list, add_params="", folder="", cwdir=os.getcwd(), suppress=True) -> str:
    name = get_name(param_list)
    if not suppress:
        print(f"Command: {get_command(exe, param_list, folder, add_params)}")
    try:
        # os.system(get_command(exe, param_list, folder, add_params))
        if(suppress):
            subprocess.run(
                [get_command(exe, param_list, folder, add_params)],
                shell=True,
                capture_output=False,
                text=False,
                cwd=cwdir,
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL
            )
        else:
            subprocess.run(
                [get_command(exe, param_list, folder, add_params)],
                shell=True,
                capture_output=False,
                text=False,
                cwd=cwdir
            )
    except:
        pass
    return name

def get_name(param_list) -> str:
    if len(param_list) < 1:
        raise Exception(
            "When adding a point in the PD, you need to give as argument a list of [name, value] lists."
        )
    name = ""
    for i, param in enumerate(param_list):
        if len(param) != 2:
            raise Exception(
                "When adding a point in the PD, you need to give as argument a list of [name, value] lists."
            )
        if i != 0:
            name = f"{name}_{param[0].split('/')[-1]}:{param[1]}"
        else:
            name = f"{param[0].split('/')[-1]}:{param[1]}"
    return name