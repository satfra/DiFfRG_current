import numpy as np
import glob
import json
import pandas
import xml.etree.ElementTree as ET
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from tqdm import tqdm
import os
from functools import partial
from multiprocessing import Pool
from scipy import optimize
from scipy.interpolate import make_interp_spline

from DiFfRG.utilities import globalize
from DiFfRG.utilities import get_all_sims

# A class to read in .pvd files
class FEMData:
    def __read_pvd(self, only_one: bool = False, at_t: int = -1, pool_size: int = 32):
        @globalize
        def load_one_step(i):
            file = self.files[i]
            t = self.timesteps[i]

            # check if the data is already loaded
            if t in self.slice_cache:
                return self.slice_cache[t]

            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(file)
            reader.Update()
            polydata = reader.GetOutput()
            nodes, data = get_vtk_data(polydata)

            # check if there are any NaNs in the data
            for sub_data in data.values():
                if np.any(np.isnan(sub_data)):
                    print("NaNs in data at t = ", t)

                    # remove this timestep from the cache
                    m_idx = int(np.argmin(np.abs(self.timesteps - t)))
                    if np.isclose(self.timesteps[m_idx], t):
                        self.timesteps = np.delete(self.timesteps, m_idx)
                        self.files.pop(m_idx)

                        new_idx = int(np.argmin(np.abs(self.timesteps - t)))
                        return load_one_step(new_idx)

            self.slice_cache[t] = {"t": t, "nodes": nodes, "point_data": data}

            return self.slice_cache[t]

        if only_one:
            if at_t < 0:
                idx = -1
            else:
                idx = np.argmin(np.abs(self.timesteps - at_t))
            return load_one_step(idx)

        pool = Pool(pool_size)
        return pool.map(load_one_step, range(len(self.files)))

    def __init__(self, filename : str):
        self.filename = filename
        self.slice_cache = {}

        if filename[-4:] != ".pvd":
            raise Exception("only .pvd files are supported!")

        self.dir = os.path.dirname(os.path.abspath(filename))
        if self.dir[-1] != "/":
            self.dir = self.dir + "/"

        self.timesteps = []
        self.files = []
        tree = ET.parse(filename)
        root = tree.getroot()
        for timestep_data in root.iter("DataSet"):
            self.timesteps.append(float(timestep_data.get("timestep")))
            self.files.append(self.dir + timestep_data.get("file"))
        self.timesteps = np.array(self.timesteps)

    def get_full_data(self):
        if not self.data:
            self.data = FEMData.__read_pvd(self.filename)
        return self.data

    def get_data_slice(self, t : float = -1):
        if t < 0:
            t = self.timesteps[-1]
        t = self.timesteps[np.argmin(np.abs(self.timesteps - t))]
        if t in self.slice_cache:
            return self.slice_cache[t]
        self.slice_cache[t] = self.__read_pvd(only_one=True, at_t=t)
        return self.slice_cache[t]

class FEMData1D(FEMData):
    def __init__(self, filename : str, cs=0, mass_name="u"):
        super().__init__(filename)

    def get_raw_x(self, t : float = -1):
        return self.get_data_slice(t)["nodes"][:, 0]
    
    def get_raw_array(self, name : str, t : float = -1):
        return self.get_data_slice(t)["point_data"][name]
    
    def get_x(self, t : float = -1):
        x = self.get_raw_x(t)
        # find all duplicates
        duplicates = np.where(np.diff(x) == 0)[0]
        # merge duplicates in x and average in y
        x = np.delete(x, duplicates)
        return x
    
    def get_array(self, name : str, t : float = -1):
        x = self.get_raw_x(t)
        y = self.get_raw_array(name, t)
        # find all duplicates
        duplicates = np.where(np.diff(x) == 0)[0]
        # merge duplicates in x and average in y
        x = np.delete(x, duplicates)
        # average in y
        y[duplicates] = (y[duplicates] + y[duplicates + 1]) / 2
        y = np.delete(y, duplicates + 1)
        return y
    
class SimulationData1D(FEMData1D):
    def __init__(self, name):
        self.pvd_file = name + ".pvd"
        super().__init__(self.pvd_file)

        self.params = json.load(open(name + ".log.json"))

        # find all associated csv files
        self.csv_files = glob.glob(name + "_*.csv")
        # load them using pandas
        self.csv_data = {}
        for f in self.csv_files:
            self.csv_data[f] = pandas.read_csv(f)

    def get_csv(self, name):
        for f in self.csv_files:
            if name in f:
                return self.csv_data[f]
        return None

def mkdir(path: str):
    """Creates a directory if it does not exist.

    Args:
        path (str): The path to the directory to be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def get_parameters_from_name(name: str) -> dict:
    if name[-1] == "/":
        raise Exception("cannot read params from folder name")
    if name[-4:] == ".pvd" or name[-4:] == ".csv":
        raw_filename = name.split("/")[-1][:-4]
    else:
        raw_filename = name.split("/")[-1]

    params = {}
    splits = raw_filename.split("_")
    # merge a split with the next one if it does not contain a "="
    i = 0
    while i < len(splits):
        if not ":" in splits[i]:
            if i == len(splits) - 1:
                splits.pop(i)
            else:
                splits[i] = splits[i] + "_" + splits[i + 1]
                splits.pop(i + 1)
        i += 1

    for p in splits:
        if len(p.split(":")) == 1:
            continue
        if not len(p.split(":")) == 2:
            raise Exception(
                "naming for file " + raw_filename + " could not be understood!"
            )
        param = p.split(":")[0]
        value = float(p.split(":")[1])
        params[param] = value
    return params


def get_vtk_data(vtkdata):
    """Utility function to extract nodes and data from a (loaded) vtk file.

    Args:
        vtkdata : The loaded vtk data.

    Returns:
        tuple: A tuple containing a numpy array of nodes and a dict with the arrays in the vtkdata.
    """
    
    nodes = vtk_to_numpy(vtkdata.GetPoints().GetData())
    data = {}

    number_of_arrays = vtkdata.GetPointData().GetNumberOfArrays()
    for i in range(number_of_arrays):
        name = vtkdata.GetPointData().GetArray(i).GetName()
        data[name] = vtk_to_numpy(vtkdata.GetPointData().GetAbstractArray(name))

    return nodes, data

def read_csv(csv, delim=",", header="infer"):
    """Reads a csv file and returns a pandas dataframe.

    Args:
        csv (str): The path to the csv file.
        delim (str, optional): Delimiter used in the csv file. Defaults to ",".
        header (str, optional): Header argument for pandas. Use None if no header exists. Defaults to "infer".

    Returns:
        pandas.DataFrame: The data in the csv file.
    """
    data = pandas.read_csv(csv, comment="#", delimiter=delim, header=header)
    return data


def read_k_csv(filename, delim=",", kName="kGeV"):
    """Reads a csv file which contains data for different values of k and returns the data split into separate csvs for each value of k.

    Args:
        filename (str): The path to the csv file.
        delim (str, optional): Delimiter used in the csv file. Defaults to ",".
        kName (str, optional): The name of the column which contains the values of k. Defaults to "kGeV".

    Returns:
        tuple: A tuple containing a list with the unique values of k and a list of pandas dataframes for each value of k.

    """

    csv = read_csv(filename, delim=delim)
    # We need to split the data into separate csvs for each value of k
    ks = csv[kName]
    k_values = np.unique(ks)
    data = []
    for k in k_values:
        mask = ks == k
        data.append(csv[mask])
    return k_values, data

def split_csv(csv, name="kGeV"):
    """Reads a csv file which contains data for different values of k and returns the data split into separate csvs for each value of k.

    Args:
        filename (str): The path to the csv file.
        delim (str, optional): Delimiter used in the csv file. Defaults to ",".
        kName (str, optional): The name of the column which contains the values of k. Defaults to "kGeV".

    Returns:
        tuple: A tuple containing a list with the unique values of k and a list of pandas dataframes for each value of k.

    """

    vs = csv[name]
    v_values = np.unique(vs)
    data = []
    for v in v_values:
        mask = vs == v
        data.append(csv[mask])
    return v_values, data