# get the smail directory on the python path
import sys, os
root_dir = os.path.abspath(os.path.join(__file__, "../../"))
sys.path.append(root_dir)

# download the data
import subprocess
import os
data_path = os.path.join(root_dir, "HEPdata")
command = f"cd {data_path} && source download"
subprocess.run(command, shell=True)

# now try to run on it
from ATLAS_data import run, get_defaults
arguments = get_defaults()
data, n_syst = run(**arguments)
save(data, arguments, n_syst, "test")
