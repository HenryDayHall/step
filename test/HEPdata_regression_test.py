# get the smail directory on the python path
import sys, os
root_dir = os.path.abspath(os.path.join(__file__, "../../"))
sys.path.append(root_dir)

# download the data
import subprocess
import os
data_path = os.path.join(root_dir, "HEPdata")

def has_downloads(data_path):
    in_dir = list(os.listdir(data_path))
    has_root = next((True for f in in_dir if f.endswith('.root')), False)
    has_txt = len([f for f in in_dir if f.endswith('.txt')]) > 1000
    print(f"Root file downloaded = {has_root}, text files downloaded = {has_txt}")
    return has_root and has_txt

if not has_downloads(data_path):
    command = f"cd {data_path} && ./download"
    subprocess.run(command, shell=True)
    assert has_downloads(data_path)

# now try to run on it
from ATLAS_data import get_defaults, run, save
arguments = get_defaults(data=data_path)
data, n_syst = run(**arguments)
save(data, arguments, n_syst, "test")
