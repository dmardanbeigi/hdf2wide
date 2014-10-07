# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 14:38:28 2014

@author: huml-rzm
"""

import os

def nabs(file_path):
    """
    Return a normalized absolute path using file_path.
    :param file_path:
    :return:
    """
    return os.path.normcase(os.path.normpath(os.path.abspath(file_path)))

# get git rev info for current working dir git repo
def get_git_local_changed():
    import subprocess
    local_repo_status = subprocess.check_output(['git', 'status'])
    return local_repo_status.find(
        'nothing to commit, working directory clean') == -1 and \
           local_repo_status.find('branch is up-to-date') == -1

def get_git_revision_hash(branch_name='HEAD'):
    import subprocess
    return subprocess.check_output(['git', 'rev-parse', branch_name])

def get_git_revision_short_hash(branch_name='HEAD'):
    import subprocess
    return subprocess.check_output(['git', 'rev-parse', '--short', branch_name])
    
def getFullOutputFolderPath(out_folder):
    output_folder_postfix = "rev_{0}".format(get_git_revision_short_hash().strip())
    if get_git_local_changed():
        output_folder_postfix = output_folder_postfix+"_UNSYNCED"
    return nabs(os.path.join(out_folder, output_folder_postfix))
    
def save_as_txt(fname, data):
    col_count = len(data.dtype.names)
    format_str = "{}\t" * col_count
    format_str = format_str[:-1] + "\n"
    
    header='#'+'\t'.join(data.dtype.names)+'\n'
    
    txtf = open(fname, 'w')
    txtf.write(header)
    for s in data.tolist():
        txtf.write(format_str.format(*s))
    txtf.close()  
    
def parseTrackerMode(eyetracker_mode):
    if eyetracker_mode == 'Binocular':
        return ['left', 'right']
    else:
        return [eyetracker_mode.split(' ')[0].lower()]