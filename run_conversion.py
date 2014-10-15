# -*- coding: utf-8 -*-
__author__ = 'Sol'

# ## CONVERSION SCRIPT SETTINGS ###
INCLUDE_TRACKERS = (
                    'dpi',
#                    'eyefollower', 
#                    'eyelink', 
#                    'eyetribe', 
#                    'hispeed1250', 
#                    'hispeed240',
#                    'red250', 
#                    'red500', 
#                    'redm', 
#                    't60xl', 
#                    'tx300', 
#                    'x2'
)

#INCLUDE_SUB = 'ALL'
INCLUDE_SUB = [3]

INPUT_FILE_ROOT = r"/media/Data/EDQ/data"
OUTPUT_FOLDER = r'/media/Data/EDQ/data_npy/'

SAVE_NPY = True
SAVE_TXT = False

#DPICAL is required to calibrate DPI
BLOCKS_TO_EXPORT = ['DPICAL', 'FS']

#In 2014 Apr-May EDQ recordings ROW_INDEX in DPICAL block is not set right.
#Set FIX_DPI_CAL to True if dealing with these recordings
FIX_DPI_CAL = True
calibrate_dpi = True

GLOB_PATH_PATTERN = INPUT_FILE_ROOT + r"/*/*/*.hdf5"
##################################

import os, sys
import glob
import re
from timeit import default_timer as getTime

import numpy as np
import matplotlib.pylab as plt
plt.ion()

import tables
from collections import OrderedDict

from constants import (MONOCULAR_EYE_SAMPLE, BINOCULAR_EYE_SAMPLE, MESSAGE,
                       MULTI_CHANNEL_ANALOG_INPUT,
                       wide_row_dtype, msg_txt_mappings,
                       dpi_cal_fix, stim_pos_mappings,
)
                       
from edq_shared import (getFullOutputFolderPath, nabs, 
                       save_as_txt, parseTrackerMode, VisualAngleCalc,
                       detect_rollingWin, #for DPI calibration 
                       filter_trackloss,
)

try:
    from yaml import load, dump
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
if sys.version_info[0] != 2 or sys.version_info[1] >= 7:
    def construct_yaml_unistr(self, node):
        return self.construct_scalar(node)

    Loader.add_constructor(u'tag:yaml.org,2002:str', construct_yaml_unistr)

#Bilateral filter from OpenCV lilbrary is used to filter DPI data
if 'dpi' in INCLUDE_TRACKERS:
    import cv2
####


OUTPUT_FOLDER = getFullOutputFolderPath(OUTPUT_FOLDER)

print 'OUTPUT_FOLDER:', OUTPUT_FOLDER


def getInfoFromPath(fpath):
    """

    :param fpath:
    :return:
    """
    if fpath.lower().endswith(".hdf5"):
        fpath, fname = os.path.split(fpath)
    return fpath.rsplit(os.path.sep, 3)[-2], np.uint(
        re.split('_|.hdf5', fname)[-2])


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


DATA_FILES = [nabs(fpath) for fpath in glob.glob(GLOB_PATH_PATTERN) if
              analyseit(fpath)]

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
                      
dpi_sample_fields = ['session_id', 'device_time', 'time',
                      'AI_4', 'AI_5', 'device_id',
                      'AI_0', 'AI_1', 'device_id',
                      'AI_2']

screen_measure_fields = ('screen_width', 'screen_height', 'eye_distance')
cv_fields = ['SESSION_ID', 'trial_id', 'TRIAL_START', 'TRIAL_END', 'posx',
             'posy', 'dt', 'ROW_INDEX', 'BLOCK']

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
        msg_event_text = msg_table.read_where("session_id == %d" % (sid))[
            'text']

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
        if calibrate_dpi & FIX_DPI_CAL & (et_model == 'dpi'):
                print 'Warning! ROW_INDEX fix for DPI calibration is ENABLED'
        # Get the condition variable set rows for the 'FS' and/or 'DPICAL' trial type
        for block in BLOCKS_TO_EXPORT:
            ecvTable = hub_file.root.data_collection.condition_variables\
                .EXP_CV_1
            cv_rows = ecvTable.read_where(
                '(BLOCK == "%s") & (SESSION_ID == %d)' % (block, session_id))
            cv_row_count = len(cv_rows)
            if cv_row_count == 0:
#                print "Skipping Session %d, not FS blocks" % (session_id)
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
                if et_model == 'dpi':
                    sample_table = getEventTableForID(hub_file, MULTI_CHANNEL_ANALOG_INPUT)
                    sample_fields = dpi_sample_fields
                else:
                    sample_table = getEventTableForID(hub_file, MONOCULAR_EYE_SAMPLE)
                
#                if sample_table.nrows == 0:
#
#                    sample_table = getEventTableForID(hub_file, BINOCULAR_EYE_SAMPLE)
#                    sample_fields = binoc_sample_fields
#                else:
#                    sample_fields = mono_sample_fields

            if et_model == 'eyetribe':
                # Use raw_x, raw_y instead of gaze
                sample_fields = [s.replace('gaze', 'raw') for s in
                                 sample_fields]
                # Data collected for eyetribe seems to have been using a
                # version of
                # script
                # that calculated the time incorrectly; so here we fix it.
                delay_col = sample_table.col('delay')[0]
                if delay_col != 0.0:
                    # fix the time and delay fields of eye tribe files;
                    # changes are
                    # saved back t hdf5
                    time_mod_count = sample_table.modify_column(0,
                                                                sample_table.nrows,
                                                                column=sample_table.col(
                                                                    'logged_time'),
                                                                colname='time')
                    delay_nod_count = sample_table.modify_column(0,
                                                                 sample_table.nrows,
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
                # selection is between cv_set['TRIAL_START'], next_cvs[
                # 'TRIAL_START']
                # set the TRIAL_END var for current row to == next_cvs[
                # 'TRIAL_START']
                # for targets 0 -(n-1)
                cv_vals = [cv_set[cvf] for cvf in cv_fields]
                
                ####Fixes ROW_INDEX in DPI calibration routine for 2014 Apr-May EDQ recordings
                if FIX_DPI_CAL & (cv_vals[-1] == 'DPICAL'):
                    
                    cv_vals[-2]=dpi_cal_fix[cv_vals[-2]]                    
                ###                
                
                tpdegxy = pix2deg(cv_vals[TARGET_POS_X_IX],
                                  cv_vals[TARGET_POS_Y_IX])
                cv_vals[trial_end_col_index] = next_cvs['TRIAL_START']

                targ_pos_samples = sample_table.where(
                    "(session_id == %d) & (time >= %.6f) & (time <= %.6f)" % (
                        cv_set['SESSION_ID'], cv_set['TRIAL_START'],
                        next_cvs['TRIAL_START']))
                for sample in targ_pos_samples:
                    sample_vals = [sample[svn] for svn in sample_fields]
                    
                    if et_model == 'dpi':
                        rdegxy = (sample_vals[RIGHT_EYE_POS_X_IX],
                                     sample_vals[RIGHT_EYE_POS_Y_IX])
                        ldegxy = (sample_vals[LEFT_EYE_POS_X_IX],
                                     sample_vals[LEFT_EYE_POS_Y_IX])
                    else:    
                        rdegxy = pix2deg(sample_vals[RIGHT_EYE_POS_X_IX],
                                         sample_vals[RIGHT_EYE_POS_Y_IX])
                        ldegxy = pix2deg(sample_vals[LEFT_EYE_POS_X_IX],
                                         sample_vals[LEFT_EYE_POS_Y_IX])
                    try:
                        sample_array_list.append(tuple(
                            session_info_vals + cv_vals + sample_vals + list(
                                tpdegxy) + list(
                                ldegxy) + list(rdegxy)))
                    except:
                        import traceback

                        traceback.print_exc()

            # process last target pos.
            cv_set = cv_rows[-1]
            cv_vals = [cv_set[cvf] for cvf in cv_fields]
            tpdegxy = pix2deg(cv_vals[TARGET_POS_X_IX],
                              cv_vals[TARGET_POS_Y_IX])
            targ_pos_samples = sample_table.where(
                "(session_id == %d) & (time >= %.6f) & (time <= %.6f)" % (
                    cv_set['SESSION_ID'], cv_set['TRIAL_START'],
                    cv_set['TRIAL_END']))
            for sample in targ_pos_samples:
                sample_vals = [sample[svn] for svn in sample_fields]
                
                if et_model == 'dpi':
                    rdegxy = (sample_vals[RIGHT_EYE_POS_X_IX],
                                 sample_vals[RIGHT_EYE_POS_Y_IX])
                    ldegxy = (sample_vals[LEFT_EYE_POS_X_IX],
                                 sample_vals[LEFT_EYE_POS_Y_IX])
                else:    
                    rdegxy = pix2deg(sample_vals[RIGHT_EYE_POS_X_IX],
                                     sample_vals[RIGHT_EYE_POS_Y_IX])
                    ldegxy = pix2deg(sample_vals[LEFT_EYE_POS_X_IX],
                                     sample_vals[LEFT_EYE_POS_Y_IX])
                try:
                    sample_array_list.append(tuple(
                        session_info_vals + cv_vals + sample_vals + list(
                            tpdegxy) + list(ldegxy) + list(
                            rdegxy)))
                except:
                    import traceback

                    traceback.print_exc()

            sample_data_by_session.append(sample_array_list)
    return sample_data_by_session, pix2deg


def getScreenMeasurements(dpath, et_model_display_configs):
    """

    :param dpath:
    :param et_model_display_configs:
    :return:
    """
    et_model, _ = getInfoFromPath(dpath)
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
              "" \
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
              "" \
              "a root.data_collection.condition_variables.EXP_CV_1 " \
              "table.\n\tSKIPPING FILE.\n<<<<<<\n" % (
                  file_path)
        if hub_file:
            hub_file.close()
            hub_file = None
        return False

    return True
    

def filter_bilateral(data, sigmaSpace=0, d=-1, sigmaColor=0):
    """
    Filters DPI data using bilateral filter
    
    @author: Raimondas Zemblys
    @email: raimondas.zemblys@humlab.lu.se
    """ 
    for _eye in ['left', 'right']:
        for _unit in ['gaze', 'angle']:
            for _dir in ['x', 'y']:
                _key = '_'.join((_eye, _unit, _dir))
                data[_key] = np.squeeze(cv2.bilateralFilter(data[_key], d,sigmaColor,sigmaSpace))
    return data

def handle_dpi_multisession(data):        
    #Finds last block of calibration and Fixate-Saccade task
    print 'Multisession DPI data found'
    sessions = np.unique(data['session_id'])
    print sessions
    _fs = []
    _cal = []
    for session_id in sessions:
        session_data = data['session_id'] == session_id
        
        DATA_SESSION = data[session_data]
        cal_block = DATA_SESSION['BLOCK'] == 'DPICAL'
        exp_block = DATA_SESSION['BLOCK'] == 'FS'
        
        _cal.append( len(np.unique(DATA_SESSION[cal_block]['ROW_INDEX'])))
        _fs.append( len(np.unique(DATA_SESSION[exp_block]['ROW_INDEX'])))
    
    _fs_LastOcc = len(_fs) - 1 - _fs[::-1].index(49)
    _cal_=_cal[:_fs_LastOcc+1]
    _cal_LastOcc = len(_cal_) - 1 - _cal_[::-1].index(25)
    
    print _fs
    print _cal
    
    print _fs_LastOcc, _cal_LastOcc
    
    session_data = data['session_id'] == sessions[_cal_LastOcc]
    DATA_CAL=data[session_data]
    cal_block = DATA_CAL['BLOCK'] == 'DPICAL'
    DATA_CAL=DATA_CAL[cal_block]
    
    
    session_data = data['session_id'] == sessions[_fs_LastOcc]
    DATA_EXP=data[session_data]
    exp_block = DATA_EXP['BLOCK'] == 'FS'
    DATA_EXP=DATA_EXP[exp_block]        
    
    return np.hstack((DATA_CAL, DATA_EXP))
    ### Handle multisession END ###

def build_polynomial(X, Y, poly_type):
    if poly_type == 'linear':
        Px = np.vstack((X, np.ones(len(X)))).T
        Py = np.vstack((Y, np.ones(len(Y)))).T

    elif poly_type == 'villanueva':
        Px = np.vstack((X**2, Y**2, X, Y, X*Y, np.ones(len(X)))).T
        Py = np.vstack((X**2, Y**2, X, Y, X*Y, np.ones(len(X)))).T

    return Px, Py 
############### MAIN RUNTIME SCRIPT ########################
#
# Below is the actual script that is run when this file is run through
# the python interpreter. The code above defines functions used by the below
# runtime script.
#

#DPI calibration config
cal_points_sets = dict([
    (5, np.array([0,4,12,20,24])), 
    (9, np.array([0,2,4,10,12,14,20,22,24])), 
    (14, np.array([1,2,35,7,9,11,12,13,15,17,19,21,22,23])),
    (16, np.array([0,6,18,24,4,8,16,20,2,7,17,22,10,11,13,14])),
    (17, np.array([0,6,12,18,24,4,8,16,20,2,7,17,22,10,11,13,14])),
    (25, np.arange(25))
])

win_select_funcs = dict([
    ('roll', detect_rollingWin)
])

calibration_settings_set = [{
    'poly_type': 'villanueva',
    'cal_point_set': 16,
    'win_select_func': 'roll'
}]

win_size=0.1                
window_skip = 0.2
wsa='fiona'  

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
            t0 = getTime()
            dpath, dfile = os.path.split(file_path)
            print "Processing file %d / %d. \r" % (
                file_proc_count + 1, total_file_count),
            screen_measurments, et_model = getScreenMeasurements(file_path,
                                                                 et_model_display_configs)

            if et_model == 'eyetribe':
                # open eyetribe files in update mode so time stamp issue can
                # be fixed in files.
                hub_file = openHubFile(dpath, dfile, 'a')
            else:
                hub_file = openHubFile(dpath, dfile, 'r')

            if not checkFileIntegrity(hub_file):
                continue

            wide_format_samples_by_session, pix2deg = convertEDQ(hub_file,
                                                        screen_measurments,
                                                        et_model)
            if wide_format_samples_by_session == None or len(
                    wide_format_samples_by_session) == 0:
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
            
            data_wide = np.array(wide_format_samples, dtype=wide_row_dtype)
            
#            #Eye selection error 
            check_eye = dict()
            if data_wide['eyetracker_mode'][0] != 'Binocular':
                eye = parseTrackerMode(data_wide['eyetracker_mode'][0])
                
                #Becomes True if all nans found                
                check_eye['right'] = sum(np.isnan(data_wide['right_gaze_x'])) == len(data_wide['right_gaze_x'])
                check_eye['left'] = sum(np.isnan(data_wide['left_gaze_x'])) == len(data_wide['left_gaze_x'])
                
                if check_eye['right'] &  check_eye['left']:
                    print "Recording does not contain any data...skipping"
                    continue
                elif check_eye[eye[0]]:
                    print "Eye selection error...correcting"
                    eye_corr = check_eye.keys()[check_eye.values().index(False)].title()
                    data_wide['eyetracker_mode'] = eye_corr+' eye' 
            
            #DPI calibration
            if calibrate_dpi & (et_model == 'dpi'):
                t1 = getTime()
                print "DPI calibration"
                
                if (len(np.unique(data_wide['session_id'])) > 1):
                    data_wide = handle_dpi_multisession(data_wide)
                
                data_wide = filter_trackloss(filter_bilateral(data_wide), et_model)
                data_wide_raw = np.copy(data_wide)

                poly_type = calibration_settings_set[0]['poly_type']
                cal_point_set = calibration_settings_set[0]['cal_point_set']
                win_select_func = calibration_settings_set[0]['win_select_func']
                
                cal_points = cal_points_sets[cal_point_set]
                cal_block = data_wide['BLOCK'] == 'DPICAL'
                exp_block = data_wide['BLOCK'] == 'FS'
                DATA_CAL = data_wide[cal_block]
#                DATA_EXP = data_wide[exp_block]
                
                args={
                      'win_size': win_size,
                      'window_skip': window_skip,
                      'wsa': ['fiona']}

                ### CALIBRATION START ###
                #TODO: deal with missing calibration points
                stim_CAL = win_select_funcs[win_select_func](DATA_CAL, **args)  
                cal_ind=stim_CAL['ROW_INDEX'] == cal_points[:, None]
                cal_ind=np.array(np.sum(cal_ind, axis=0), dtype=bool)
                
                #TODO: Handle missing calibration points: replace with random ones
                units = 'gaze'
                for eye in  parseTrackerMode(data_wide['eyetracker_mode'][0]):               
#                    plt.figure()
                    Px, Py = build_polynomial(stim_CAL['_'.join((eye, units, 'fix_x'))][cal_ind], 
                                              stim_CAL['_'.join((eye, units, 'fix_y'))][cal_ind], poly_type)
                    
                    calX, calY = np.linalg.lstsq(Px, stim_CAL[stim_pos_mappings[units]+'x'][cal_ind])[0], \
                                 np.linalg.lstsq(Py, stim_CAL[stim_pos_mappings[units]+'y'][cal_ind])[0]
                    
                    Px_data, Py_data = build_polynomial(data_wide_raw['_'.join((eye, units, 'x'))], 
                                                        data_wide_raw['_'.join((eye, units, 'y'))] , poly_type)
                    
#                    plt.plot(data_wide_raw['_'.join((eye, units, 'x'))])
#                    plt.plot(data_wide[stim_pos_mappings[units]+'x'])
                    data_wide['_'.join((eye, units, 'x'))] = np.dot(Px_data, calX)
                    data_wide['_'.join((eye, units, 'y'))] = np.dot(Py_data, calY)
                    
                    plt.plot(data_wide['_'.join((eye, units, 'x'))])

                    (data_wide['_'.join((eye, 'angle', 'x'))], 
                     data_wide['_'.join((eye, 'angle', 'y'))])=pix2deg(data_wide['_'.join((eye, units, 'x'))],
                                                                       data_wide['_'.join((eye, units, 'y'))])
                                        
                ### CALIBRATION END ###        
                print 'DPI calibration duration: ', getTime()-t1     
                
            #TODO: deal with multisession recordings
            #      filter trackloss 
            
            print 'Conversion duration: ', getTime()-t0

            #Save
            if SAVE_NPY:
                et_dir = nabs(r"%s/%s" % (OUTPUT_FOLDER, et_model))
                if not os.path.exists(et_dir):
                    os.mkdir(et_dir)
                np_file_name = r"%s/%s_%s.npy" % (
                    et_dir, et_model, dfile[:-5])
                
                t0 = getTime()
                np.save(np_file_name, data_wide)
                print 'RAW_NPY save duration: ', getTime()-t0
            
            if SAVE_TXT:
                et_dir = nabs(r"%s/%s" % (OUTPUT_FOLDER, et_model))
                if not os.path.exists(et_dir):
                    os.mkdir(et_dir)
                txt_file_name = r"%s/%s_%s.txt" % (
                    et_dir, et_model, dfile[:-5])
                
                t0 = getTime()
                save_as_txt(txt_file_name, data_wide)
                print 'RAW_TXT save duration: ', getTime()-t0

            hub_file.close()
            print 
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
