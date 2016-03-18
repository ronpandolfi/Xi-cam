# -*- coding: utf-8 -*-
"""
@author: lbluque
"""
from subprocess import Popen


def run_local_script(run_cmd, script_path):
    proc = Popen(['xterm', '-T', 'SPEW Reconstruction job', '-hold', '-e', run_cmd, script_path])
    # TODO Write while loop to keep track of running
    # TODO think of how to connect stoud/stderr pipe to gui job description


def run_nersc_batch_script(wrapper_path, system, newt_client):
    cmd = 'sbatch '
    newt_client.execute_command()
    return
