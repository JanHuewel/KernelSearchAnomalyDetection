from os.path import exists
import os
import subprocess
import shutil
import numpy as np
from itertools import product
list_of_data = ["data/dd_test_basic_anomaly2.csv"]
list_of_segment_lengths = ["100"]
list_of_methods = ["cov", "likelihood", "MSE", "KLD", "sampling", "sampling2"]
list_of_clusterings = ["PIC", "Agg"]
list_of_normalizations = ["0", '1', '2']
list_of_ground_truths = [["0", "0", "0", "0", "1", "0", "0", "0", "0", "0"]]

config = product(list_of_data, list_of_segment_lengths, list_of_methods,
                 list_of_clusterings, list_of_normalizations)

#test_config = ["data/dd_test_basic_anomaly2.csv", "100", "cov", "PIC", "0"]
#popen = subprocess.Popen(, stdout=subprocess.PIPE, universal_newlines=True)
cmd = ['python', 'Main.py']


def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        data_split = cmd[2].split("/")
        if len(data_split) == 1:
            output_path = "Results/" + "_".join(cmd[2:])
        else:
            output_path = "Results/" + str(data_split[-1]) + "_".join(conf[3:])
        out_file = open(output_path + ".txt", "w")
        out_file.write("ERROR")
        # raise subprocess.CalledProcessError(return_code, cmd)


for i in range(10):
    for path in execute(cmd):
        pass
    folder = "Results/"
    shutil.make_archive(f"trial_{i}", 'zip', os.path.join(os.getcwd(), folder))
