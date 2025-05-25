import glob
import os
import shutil
from multiprocessing import Pool
import subprocess

from DiFfRG.utilities import globalize
import DiFfRG.file_io as io

def get_command(exe, param_list, folder="", add_params="") -> str:
    if len(param_list) < 1:
        raise Exception(
            "When adding a point in the PD, you need to give as argument a list of [name, value] lists."
        )
    name = ""
    attach = ""
    for i, param in enumerate(param_list):
        if len(param) != 2:
            raise Exception(
                "When adding a point in the PD, you need to give as argument a list of [name, value] lists."
            )
        if i != 0:
            name = f"{name}_{param[0].split('/')[-1]}:{param[1]}"
        else:
            name = f"{param[0].split('/')[-1]}:{param[1]}"
        attach = f"{attach} -sd {param[0]}={param[1]}"
    if folder != "":
        return f"{exe} -ss /output/name={name} {attach} -ss /output/folder={folder} {add_params}"
    return f"{exe} -o {name} {attach} {add_params}"


def point_exists(param_list, folder="") -> bool:
    files_pvd = glob.glob(folder + "*.pvd")
    name = get_name(param_list)
    for file in files_pvd:
        file_name = file.split('/')[-1][:-4]
        if name in file_name:
            if not sim_finished(name, folder):
                print("Simulation not finished, but pvd file exists! Restarting.")
                return False
            return True
    return False


def sim_finished(name, folder="") -> bool:
    log_file = folder + name + ".log"
    if os.path.isfile(log_file):
        with open(log_file, "r") as f:
            lines = f.readlines()
            if len(lines) < 1:
                return False
            if "finish" in lines[-1]:
                return True
            return False


def point_finished(param_list, folder="") -> bool:
    name = get_name(param_list)
    return sim_finished(name, folder)


def run_point(exe, param_list, add_params="", folder="", cwdir=os.getcwd(), suppress=True) -> str:
    name = get_name(param_list)
    if point_exists(param_list, folder):
        return name
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


def get_unfinished_sims(folder, pool_size=16) -> list:
    sims = []
    files_pvd = glob.glob(folder + "*.pvd")

    @globalize
    def should_remove(file):
        file_name = file.split('/')[-1][:-4]
        if not sim_finished(file_name, folder):
            return file
        return None

    pool = Pool(pool_size)
    sims = pool.map(should_remove, files_pvd)
    sims = [f for f in sims if f != None]
    return sims


def clean_unfinished_sims(folder):
    removal_list = get_unfinished_sims(folder)
    for r in removal_list:
        base_name = r[:-4]
        files = glob.glob(base_name + "*.*")
        folder = base_name + "/"
        print(f"Removing {folder}")
        try:
            shutil.rmtree(folder)
        except:
            print(f"Could not remove {folder}")
        print(f"Removing files {files}")
        for file in files:
            try:
                os.remove(file)
            except:
                print(f"Could not remove {file}")


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