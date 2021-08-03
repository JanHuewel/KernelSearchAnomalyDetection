"""
Call Main.py for a fixed number of times and save the trials in separate archives.
"""
import os
import subprocess
import shutil

cmd = ['python', 'Main.py']


def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


for i in range(0, 23):
    for path in execute(cmd):
        pass
    folder = "Results/"
    shutil.make_archive(f"trial_{i}", 'zip', os.path.join(os.getcwd(), folder))
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))




