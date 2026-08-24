import h5py

def get_config_from_hdf5(file):
    """Reads the config from an hdf5 file and returns it as a nested dictionary.

    Args:
        file (str): The path to the hdf5 file."""
    
    # the config is stored at /config/ - groups are nested dictionaries with attributes.
    with h5py.File(file, "r") as f:
        config = {}
        # iterate over the groups in /config/
        for group in f["config"]:
            config[group] = {}
            for key, value in f["config"][group].attrs.items():
                config[group][key] = value
    return config

def get_fem_from_hdf5(file):
    """Reads the fem data from an hdf5 file and returns it as a nested dictionary.

    Args:
        file (str): The path to the hdf5 file."""
    
    # the fem data is stored at /FE/ - groups are nested dictionaries with attributes.
    with h5py.File(file, "r") as f:
        fem = []
        # the groups in /FE/ are named 000000, 000001, etc. - we need to sort them by name to get the correct order.
        for i, group in enumerate(sorted(f["FE"], key=lambda x: int(x))):
            fem.append({})
            for key, value in f["FE"][group].attrs.items():
                fem[-1][key] = value
            # Get all datasets in the group. We need to deduce their shapes (they may be 1D, 2D or 3D)
            for dataset in f["FE"][group]:
                fem[-1][dataset] = f["FE"][group][dataset][()]
        
    return fem

def get_scalars_from_hdf5(file):
    """Reads the scalars from an hdf5 file and returns them as a dictionary.

    Args:
        file (str): The path to the hdf5 file."""
    
    # the scalars are stored at /scalars/ - they're all stored as datasets
    with h5py.File(file, "r") as f:
        scalars = {}
        for key, value in f["scalars"].items():
            scalars[key] = value[()]
    return scalars

class SimulationData:
    def __init__(self, name):
        self.hdf5_file = name

        self.params = get_config_from_hdf5(self.hdf5_file)

        self.fem_data = get_fem_from_hdf5(self.hdf5_file)

        self.scalars = get_scalars_from_hdf5(self.hdf5_file)