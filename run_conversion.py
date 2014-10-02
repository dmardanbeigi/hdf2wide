# -*- coding: utf-8 -*-
__author__ = 'Sol'

# ## CONVERSION SCRIPT SETTINGS ###
SAVE_NPY = True
SAVE_TXT = False
OUTPUT_FOLDER = r'/output/EDQ/hdf2wide/'
INPUT_FILE_ROOT = r"/media/Data/EDQ/data"

INCLUDE_TRACKERS = ('eyefollower', 'eyelink', 'eyetribe', 'hispeed1250', 'hispeed240',
              'red250', 'red500', 'redm', 't60xl', 'tx300', 'x2')
INCLUDE_TRACKERS = ('hispeed1250')


INCLUDE_SUB = 'ALL'
#INCLUDE_SUB = [127]

BLOCKS_TO_EXPORT = ['FS']


GLOB_PATH_PATTERN = INPUT_FILE_ROOT+r"/*/*/*.hdf5"
##################################

import sys, os
from timeit import default_timer as getTime
from constants import (MONOCULAR_EYE_SAMPLE, BINOCULAR_EYE_SAMPLE, MESSAGE,
                       et_nan_values, wide_row_dtype, msg_txt_mappings)
from pix2deg import VisualAngleCalc
import tables
from collections import OrderedDict
import glob
import re
import numpy as np

try:
    from yaml import load, dump
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
if sys.version_info[0] != 2 or sys.version_info[1] >= 7:
    def construct_yaml_unistr(self, node):
        return self.construct_scalar(node)

    Loader.add_constructor(u'tag:yaml.org,2002:str', construct_yaml_unistr)

# get git rev info for current working dir git repo
def get_git_revision_hash(branch_name='HEAD'):
    import subprocess
    return subprocess.check_output(['git', 'rev-parse', branch_name])

def get_git_revision_short_hash(branch_name='HEAD'):
    import subprocess
    return subprocess.check_output(['git', 'rev-parse', '--short', branch_name])

def get_git_local_changed():
    import subprocess
    local_repo_status = subprocess.check_output(['git', 'status'])
    print "======="
    print local_repo_status
    print '=========='
    return local_repo_status.find('branch is up-to-date') == -1
    
def nabs(file_path):
    """
    Return a normalized absolute path using file_path.
    :param file_path:
    :return:
    """
    return os.path.normcase(os.path.normpath(os.path.abspath(file_path)))

print 'get_git_revision_hash:',get_git_revision_hash()
print 'get_git_revision_short_hash:',get_git_revision_short_hash()
print 'get_git_local_changed:',get_git_local_changed()

OUTPUT_FOLDER = nabs(os.path.join(OUTPUT_FOLDER,get_git_revision_short_hash()))

print 'OUTPUT_FOLDER:',OUTPUT_FOLDER

def getTrackerTypeFromPath(fpath):
    """

    :param fpath:
    :return:
    """
    if fpath.lower().endswith(".hdf5"):
        fpath, _ = os.path.split(fpath)
    return fpath.rsplit(os.path.sep, 3)[-2]

def getInfoFromPath(fpath):
    """

    :param fpath:
    :return:
    """
    if fpath.lower().endswith(".hdf5"):
        fpath, fname = os.path.split(fpath)
    return fpath.rsplit(os.path.sep, 3)[-2], np.uint(re.split('_|.hdf5', fname)[-2])

def keepit(fpath):
    """

    :param fpath:
    :return:
    """
    tracker_type = getTrackerTypeFromPath(fpath)
    return tracker_type not in SKIP_TRACKERS

def analyseit(fpath):
    """

    :param fpath:
    :return:
    """
    tracker_type, sub = getInfoFromPath(fpath)
    if INCLUDE_SUB == 'ALL':
        return (tracker_type in INCLUDE_TRACKERS)
    else:
        return (tracker_type in INCLUDE_TRACKERS) & (sub in INCLUDE_SUB)


#DATA_FILES = [nabs(fpath) for fpath in glob.glob(GLOB_PATH_PATTERN) if
#              keepit(fpath)]
              
DATA_FILES = [nabs(fpath) for fpath in glob.glob(GLOB_PATH_PATTERN) if analyseit(fpath)]


binoc_sample_fields = ['session_id', 'device_time', 'time',
                       'left_gaze_x', 'left_gaze_y', 'left_pupil_measure1',
                       'right_gaze_x', 'right_gaze_y', 'right_pupil_measure1',
                       'status']
LEFT_EYE_POS_X_IX = binoc_sample_fields.index('left_gaze_x')
LEFT_EYE_POS_Y_IX = binoc_sample_fields.index('left_gaze_y')
RIGHT_EYE_POS_X_IX = binoc_sample_fields.index('right_gaze_x')
RIGHT_EYE_POS_Y_IX = binoc_sample_fields.index('right_gaze_y')

mono_sample_fields = ['session_id', 'device_time', 'time',
                      'gaze_x', 'gaze_y', 'pupil_measure1',
                      'gaze_x', 'gaze_y', 'pupil_measure1',
                      'status']

screen_measure_fields = ('screen_width', 'screen_height', 'eye_distance')
cv_fields = ['SESSION_ID', 'trial_id', 'TRIAL_START', 'TRIAL_END', 'posx', 'posy', 'dt',
             'ROW_INDEX', 'BLOCK']

TARGET_POS_X_IX = cv_fields.index('posx')
TARGET_POS_Y_IX = cv_fields.index('posy')

def openHubFile(filepath, filename, mode):
    """
    Open an HDF5 DataStore file.
    """
    hubFile = tables.openFile(os.path.join(filepath, filename), mode)
    return hubFile


def getEventTableForID(hub_file, event_type):
    """
    Return the pytables event table for the given EventConstant event type
    :param hub_file: pytables hdf5 file
    :param event_type: int
    :return:
    """
    evt_table_mapping = hub_file.root.class_table_mapping.read_where(
        'class_id == %d' % (event_type))
    return hub_file.getNode(evt_table_mapping[0]['table_path'])


def num(s):
    """

    :param s:
    :return:
    """
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def getSessionDataFromMsgEvents(hub_file):
    """
    Get all the session data that was saved as messages and not in the
    session meta data table
    :param hub_file:
    :return: dict
    """
    # and not in the session meta data table
    msg_table = getEventTableForID(hub_file, MESSAGE)
    # < 20 msg's written in this exp, so just read them all
    session_ids = np.unique(msg_table.read()['session_id'])
    session_infos = OrderedDict()
    for sid in session_ids:
        session_info = OrderedDict()
        session_infos[sid] = session_info
        msg_event_text = msg_table.read_where("session_id == %d"%(sid))['text']

        _msg_event_dict = dict()
        for msg in msg_event_text:
            msplit = msg.split(':')
            _msg_event_dict[msplit[0].strip()] = [t.strip() for t in msplit[1:]]

        # Parse out (painfully) the data of interest
        for org_title, txt_title in msg_txt_mappings.iteritems():
            msg_data = _msg_event_dict.get(org_title)
            if msg_data:
                if len(msg_data) == 1:
                    msg_data = msg_data[0]
                    session_info[txt_title] = num(msg_data)
                elif org_title == 'Stimulus Screen ID':
                    pstr = msg_data[1].split(',')[0:2]
                    session_info[txt_title] = num(pstr[0][1:])
                    session_info[msg_txt_mappings['Stimulus Screen ID2']] = num(
                        pstr[1][:-1])

    return session_infos


def convertEDQ(hub_file, screen_measures, et_model):
    """

    :param hub_file:
    :param screen_measures:
    :param et_model:
    :return:
    """
    display_size_mm = screen_measures['screen_width'], screen_measures[
        'screen_height']

    sample_data_by_session = []
    session_info_dict = getSessionDataFromMsgEvents(hub_file)

    for session_id, session_info in session_info_dict.items():
        # Get the condition variable set rows for the 'FS' trial type
        for block in BLOCKS_TO_EXPORT:
            ecvTable = hub_file.root.data_collection.condition_variables.EXP_CV_1
            cv_rows = ecvTable.read_where('(BLOCK == "%s") & (SESSION_ID == %d)'%(block,session_id))
            cv_row_count = len(cv_rows)
            if cv_row_count == 0:
                print "Skipping Session %d, not FS blocks"%(session_id)
                continue
    
            display_size_pix = session_info['display_width_pix'], session_info[
                'display_height_pix']
    
            pix2deg = VisualAngleCalc(display_size_mm, display_size_pix,
                                      screen_measures['eye_distance']).pix2deg
    
            session_info.update(screen_measures)
            session_info_vals = session_info.values()
    
            tracking_eye = session_info['eyetracker_mode']
            # Get the eye sample table
            if tracking_eye == 'Binocular':
                sample_table = getEventTableForID(hub_file,
                                                  BINOCULAR_EYE_SAMPLE)
                if sample_table.nrows == 0:
                    sample_table = getEventTableForID(hub_file,
                                                      MONOCULAR_EYE_SAMPLE)
                    sample_fields = mono_sample_fields
                else:
                    sample_fields = binoc_sample_fields
            else:
                sample_table = getEventTableForID(hub_file,
                                                  MONOCULAR_EYE_SAMPLE)
                if sample_table.nrows == 0:
                    sample_table = getEventTableForID(hub_file,
                                                      BINOCULAR_EYE_SAMPLE)
                    sample_fields = binoc_sample_fields
                else:
                    sample_fields = mono_sample_fields
    
            if et_model == 'eyetribe':
                # Use raw_x, raw_y instead of gaze
                sample_fields = [s.replace('gaze','raw') for s in sample_fields]
                # Data collected for eyetribe seems to have been using a version of
                # script
                # that calculated the time incorrectly; so here we fix it.
                delay_col = sample_table.col('delay')[0]
                if delay_col != 0.0:
                    # fix the time and delay fields of eye tribe files; changes are
                    # saved back t hdf5
                    time_mod_count = sample_table.modify_column(0, sample_table.nrows,
                                                                column=sample_table.col(
                                                                    'logged_time'),
                                                                colname='time')
                    delay_nod_count = sample_table.modify_column(0, sample_table.nrows,
                                                                 column=sample_table.col(
                                                                     'left_gaze_z'),
                                                                 colname='delay')
    
            # create wide format txt output
            trial_end_col_index = cv_fields.index('TRIAL_END')
            sample_array_list = []
    
            for row_index, cv_set in enumerate(cv_rows[:-1]):
                assert session_id == cv_set['SESSION_ID']
                next_cvs = cv_rows[row_index + 1]
                # Get current condition var value str. Since sample time period
                # selection is between cv_set['TRIAL_START'], next_cvs['TRIAL_START']
                # set the TRIAL_END var for current row to == next_cvs['TRIAL_START']
                # for targets 0 -(n-1)
                cv_vals = [cv_set[cvf] for cvf in cv_fields]
                tpdegxy = pix2deg(cv_vals[TARGET_POS_X_IX], cv_vals[TARGET_POS_Y_IX])
                cv_vals[trial_end_col_index] = next_cvs['TRIAL_START']
    
                targ_pos_samples = sample_table.where(
                    "(session_id == %d) & (time >= %.6f) & (time <= %.6f)" % (
                    cv_set['SESSION_ID'], cv_set['TRIAL_START'],
                    next_cvs['TRIAL_START']))
                for sample in targ_pos_samples:
                    sample_vals = [sample[svn] for svn in sample_fields]
    
                    rdegxy = pix2deg(sample_vals[RIGHT_EYE_POS_X_IX], sample_vals[RIGHT_EYE_POS_Y_IX])
                    ldegxy = pix2deg(sample_vals[LEFT_EYE_POS_X_IX], sample_vals[LEFT_EYE_POS_Y_IX])
                    try:
                        sample_array_list.append(tuple(
                            session_info_vals + cv_vals + sample_vals + list(tpdegxy) + list(
                                ldegxy) + list(rdegxy)))
                    except:
                        import traceback
    
                        traceback.print_exc()
    
            # process last target pos.
            cv_set = cv_rows[-1]
            cv_vals = [cv_set[cvf] for cvf in cv_fields]
            tpdegxy = pix2deg(cv_vals[3], cv_vals[4])
            targ_pos_samples = sample_table.where(
                "(session_id == %d) & (time >= %.6f) & (time <= %.6f)" % (
                cv_set['SESSION_ID'], cv_set['TRIAL_START'], cv_set['TRIAL_END']))
            for sample in targ_pos_samples:
                sample_vals = [sample[svn] for svn in sample_fields]
                rdegxy = pix2deg(sample_vals[RIGHT_EYE_POS_X_IX], sample_vals[RIGHT_EYE_POS_Y_IX])
                ldegxy = pix2deg(sample_vals[LEFT_EYE_POS_X_IX], sample_vals[LEFT_EYE_POS_Y_IX])
                try:
                    sample_array_list.append(tuple(
                        session_info_vals + cv_vals + sample_vals + list(tpdegxy) + list(ldegxy) + list(
                            rdegxy)))
                except:
                    import traceback
    
                    traceback.print_exc()
    
            sample_data_by_session.append(sample_array_list)
    return sample_data_by_session

def getScreenMeasurements(dpath, et_model_display_configs):
    """

    :param dpath:
    :param et_model_display_configs:
    :return:
    """
    et_model = getTrackerTypeFromPath(dpath)
    display_param = et_model_display_configs.get(et_model)
    if display_param is None:
        et_config_path = glob.glob('./configs/*%s.yaml' % (et_model))
        if et_config_path:
            et_config_path = nabs(et_config_path[0])
            display_config = load(file(et_config_path, 'r'), Loader=Loader)
            dev_list = display_config.get('monitor_devices')
            for d in dev_list:
                if d.keys()[0] == 'Display':
                    d = d['Display']
                    width = d.get('physical_dimensions', {}).get('width')
                    height = d.get('physical_dimensions', {}).get('height')
                    eye_dist = d.get('default_eye_distance', {}).get(
                        'surface_center')
                    et_model_display_configs[et_model] = OrderedDict()
                    et_model_display_configs[et_model][
                        screen_measure_fields[0]] = width
                    et_model_display_configs[et_model][
                        screen_measure_fields[1]] = height
                    et_model_display_configs[et_model][
                        screen_measure_fields[2]] = eye_dist
                    return et_model_display_configs[et_model], et_model
    return display_param, et_model


def checkFileIntegrity(hub_file):
    """

    :param hub_file:
    :return:
    """
    try:
        tm = hub_file.root.class_table_mapping
    except:
        print "\n>>>>>>\nERROR processing Hdf5 file: %s\n\tFile does not have " \
              "a root.class_table_mapping table.\n\tSKIPPING FILE.\n<<<<<<\n" \
              % (
        file_path)
        if hub_file:
            hub_file.close()
            hub_file = None
        return False
    try:
        tm = hub_file.root.data_collection.condition_variables.EXP_CV_1
    except:
        print "\n>>>>>>\nERROR processing Hdf5 file: %s\n\tFile does not have " \
              "a root.data_collection.condition_variables.EXP_CV_1 " \
              "table.\n\tSKIPPING FILE.\n<<<<<<\n" % (
        file_path)
        if hub_file:
            hub_file.close()
            hub_file = None
        return False

    return True

############### MAIN RUNTIME SCRIPT ########################
#
# Below is the actual script that is run when this file is run through
# the python interpreter. The code above defines functions used by the below
# runtime script.
#
if __name__ == '__main__':
    try:
        et_model_display_configs = dict()
        scount = 0

        start_time = getTime()

        if not os.path.exists(OUTPUT_FOLDER):
            os.mkdir(OUTPUT_FOLDER)

        col_count = len(wide_row_dtype.names)
        format_str = "{}\t" * col_count
        format_str = format_str[:-1] + "\n"
        row_names = wide_row_dtype.names
        header_line = '\t'.join(row_names) + '\n'
        file_proc_count = 0
        total_file_count = len(DATA_FILES)
        hub_file = None
        for file_path in DATA_FILES:
            dpath, dfile = os.path.split(file_path)
            print "Processing file %d / %d. \r" % (
            file_proc_count + 1, total_file_count),
            screen_measurments, et_model = getScreenMeasurements(dpath,
                                                                 et_model_display_configs)

            if et_model == 'eyetribe':
                # open eyetribe files in update mode so time stamp issue can
                # be fixed in files.
                hub_file = openHubFile(dpath, dfile, 'a')
            else:
                hub_file = openHubFile(dpath, dfile, 'r')

            if not checkFileIntegrity(hub_file):
                continue

            wide_format_samples_by_session = convertEDQ(hub_file, screen_measurments,
                                             et_model)
            if wide_format_samples_by_session == None or len(wide_format_samples_by_session) == 0:
                print "\n>>>>>>\nERROR processing Hdf5 file: %s\n\tFile has " \
                      "no 'FS' BLOCK COND VARS.\n\tSKIPPING FILE.\n<<<<<<\n" % (
                file_path)
                if hub_file:
                    hub_file.close()
                    hub_file = None
                continue

            file_proc_count += 1
            wide_format_samples = []
            for output_samples in wide_format_samples_by_session:
                wide_format_samples.extend(output_samples)

            scount += len(wide_format_samples)
            if SAVE_NPY:
                et_dir = nabs(r"%s/%s"%(OUTPUT_FOLDER, et_model))
                if not os.path.exists(et_dir):
                    os.mkdir(et_dir)
                np_file_name = r"%s/%s_%s.npy" % (
                et_dir, et_model, dfile[:-5])
                #print 'Saving output: ',np_file_name
                np.save(np_file_name,
                        np.array(wide_format_samples, dtype=wide_row_dtype))
            if SAVE_TXT:
                et_dir = nabs(r"%s/%s"%(OUTPUT_FOLDER, et_model))
                if not os.path.exists(et_dir):
                    os.mkdir(et_dir)
                txt_file_name = r"%s/%s_%s.txt" % (
                et_dir, et_model, dfile[:-5])
                #print 'Saving output: ',txt_file_name
                txtf = open(txt_file_name, 'w')
                txtf.write(header_line)
                for s in wide_format_samples:
                    txtf.write(format_str.format(*s))
                txtf.close()

            hub_file.close()
        end_time = getTime()

        print
        print 'Processed File Count:', file_proc_count
        print 'Total Samples Selected for Output:', scount
        print "Total Run Time:", (end_time - start_time)
        print "Samples / Second:", scount / (end_time - start_time)

    except Exception, e:
        import traceback

        traceback.print_exc()
    finally:
        if hub_file:
            hub_file.close()
            hub_file = None
