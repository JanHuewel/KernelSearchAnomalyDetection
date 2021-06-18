import subprocess
import numpy as np
list_of_data = ["data/dd_test_basic_anomaly2.csv"]
list_of_segment_lengths = ["100"]
list_of_methods = ["cov"] #"likelihood", "MSE", "KLD", "sampling", "sampling2"]
list_of_clusterings = ["PIC", "Agg"]
list_of_normalizations = ["0",'1','2']
list_of_ground_truths = [["0","0","0","0","1","0","0","0","0","0"]]

data_gt_pairs = zip(list_of_data, list_of_ground_truths)

config = product(data_gt_pairs[0], list_of_segment_lengths, list_of_methods, list_of_clusterings, list_of_normalizations, data_gt_pairs[1])

cmd = ['python', 'Main.py', config]
# todo, insert the new cfg file and the folder for the .data file
#popen = subprocess.Popen(, stdout=subprocess.PIPE, universal_newlines=True)
def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

# Example
for conf in config:
    for path in execute(conf):
    print(path, end="")
    pass

    #popen.kill() #TODO does this work?

