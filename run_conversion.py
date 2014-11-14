# -*- coding: utf-8 -*-
__authors__ = 'Sol', 'r-zemblys'

### CONVERSION SCRIPT SETTINGS ###
INCLUDE_TRACKERS = (
                    'dpi',
                    'eyefollower', 
                    'eyelink', 
                    'eyetribe', 
                    'hispeed1250', 
                    'hispeed240',
                    'red250', 
                    'red500', 
                    'redm', 
                    't60xl', 
                    'tx300', 
                    'x2'
)

MONOCULAR_TRACKERS = (
                    'dpi',
#                    'eyefollower', 
                    'eyelink', 
#                    'eyetribe', 
                    'hispeed1250', 
                    'hispeed240',
#                    'red250', 
#                    'red500', 
                    'redm', 
#                    't60xl', 
#                    'tx300', 
#                    'x2'
)

INCLUDE_SUB = 'ALL'

INPUT_FILE_ROOT = r"/media/Data/EDQ/data_hdf5"
OUTPUT_FOLDER = r'/media/Data/EDQ/data_npy'

SAVE_NPY = True
SAVE_TXT = False

#DPICAL is required to calibrate DPI
BLOCKS_TO_EXPORT = ['DPICAL', 'FS']

CALIBRATE_DPI = True

#Check for binocular data averaging
CHECK_BDA = True

DATASET = 'EDQ_LUND'

PLOT_TRACKERS = (
    'dpi',
    'eyefollower', 
    'eyelink', 
    'eyetribe', 
    'hispeed1250', 
    'hispeed240',
    'red250', 
    'red500', 
    'redm', 
    't60xl', 
    'tx300', 
    'x2'
)
plot_multisession = True

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
                       smc_dtype
)
                       
from edq_shared import (getFullOutputFolderPath, nabs, 
                       save_as_txt, parseTrackerMode, VisualAngleCalc,
                       filter_trackloss,
                       plot_data,
)

if 'dpi' in INCLUDE_TRACKERS:
    import cv2 #Bilateral filter from OpenCV library is used to filter DPI data
    from edq_shared import detect_rollingWin

try:
    from yaml import load, dump
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
if sys.version_info[0] != 2 or sys.version_info[1] >= 7:
    def construct_yaml_unistr(self, node):
        return self.construct_scalar(node)

    Loader.add_constructor(u'tag:yaml.org,2002:str', construct_yaml_unistr)

####

### Conversion variables
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

FIX_DPI_CAL = False
special_multisession_cases = []

###

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
            
            if msg.find('ioHub Experiment started') > -1:
                _msg_event_dict['exp_date'] = [msg.strip('ioHub Experiment started')]
        
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
    
    if CALIBRATE_DPI & FIX_DPI_CAL & (et_model == 'dpi'):
        print 'Warning! ROW_INDEX fix for DPI calibration is ENABLED'

    for session_id, session_info in session_info_dict.items():
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
                
                    if sample_table.nrows == 0:
    
                        sample_table = getEventTableForID(hub_file, BINOCULAR_EYE_SAMPLE)
                        sample_fields = binoc_sample_fields
                    else:
                        sample_fields = mono_sample_fields

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
    return sample_data_by_session


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
    
    session_data = data['session_id'] == sessions[_cal_LastOcc]
    DATA_CAL=data[session_data]
    cal_block = DATA_CAL['BLOCK'] == 'DPICAL'
    DATA_CAL=DATA_CAL[cal_block]
    
    session_data = data['session_id'] == sessions[_fs_LastOcc]
    DATA_EXP=data[session_data]
    exp_block = DATA_EXP['BLOCK'] == 'FS'
    DATA_EXP=DATA_EXP[exp_block]        
    
    return np.hstack((DATA_CAL, DATA_EXP)), sessions[_cal_LastOcc], sessions[_fs_LastOcc]
    ### Handle multisession END ###

def build_polynomial(X, Y, poly_type):
    if poly_type == 'linear':
        Px = np.vstack((X, np.ones(len(X)))).T
        Py = np.vstack((Y, np.ones(len(Y)))).T

    elif poly_type == 'villanueva':
        Px = np.vstack((X**2, Y**2, X, Y, X*Y, np.ones(len(X)))).T
        Py = np.vstack((X**2, Y**2, X, Y, X*Y, np.ones(len(X)))).T

    return Px, Py 

#Custom exception
class ConversionError(RuntimeError):
   def __init__(self, arg):
      self.args = arg
      self.message = arg
      
def nan_equal(a,b):
    try:
        np.testing.assert_equal(a,b)
    except AssertionError:
        return False
    return True

    
#Custom conversion settings
if DATASET == 'EDQ_LUND':
    '''    
    In 2014 Apr-May EDQ recordings ROW_INDEX in DPICAL block is not set right.
    Set FIX_DPI_CAL to True if dealing with these recordings
    '''
    FIX_DPI_CAL = True
    
    '''
    Manual multisession handler. 
    Some recordings contain data from several participants 
    due to mistake when setting subject id. 
    *special_multisession_cases* array can be used to manualy couple session id
    with subject id.
    Format:
        ('subject_id', np.uint8)
        ('eyetracker_model', str, 32)
        ('session_id', np.uint8)

    '''  
    special_multisession_cases.append((121, 'eyelink', 1))
    special_multisession_cases.append((124, 'eyelink', 2))
    special_multisession_cases.append((91, 'red250', 1))
    special_multisession_cases.append((95, 'red250', 2))
    special_multisession_cases.append((84, 'red500', 1))
    special_multisession_cases.append((87, 'red500', 2))

special_multisession_cases=np.array(special_multisession_cases, dtype=smc_dtype)

#DPI calibration config
if 'dpi' in INCLUDE_TRACKERS: 
    cal_point_sets = dict([
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
        'min_cal_points': 8,
        'win_select_func': 'roll',
        'win_size': 0.175,  
        'win_type': 'sample',     
        'window_skip': 0.2,
        'wsa': 'fiona' ,
        'units': 'gaze',
    }]

OUTPUT_FOLDER = getFullOutputFolderPath(OUTPUT_FOLDER)
print 'OUTPUT_FOLDER:', OUTPUT_FOLDER

DATA_FILES = [nabs(fpath) for fpath in glob.glob(GLOB_PATH_PATTERN) if
              analyseit(fpath)]

#Check for dublicates
sub_dict = dict()
for et_model in INCLUDE_TRACKERS:
    sub_dict[et_model] = []
    for file_path in DATA_FILES:
        et, sub = getInfoFromPath(file_path)
        if et==et_model:
            sub_dict[et_model].append(sub)
    if len(sub_dict[et_model]) != len(np.unique(sub_dict[et_model])):
        unique_ids, unique_counts = np.unique(sub_dict[et_model], return_counts=True)
        
        print 'Dublicate subject ids in %s:'%et_model, \
        unique_ids[np.argwhere(unique_counts > 1).flatten()]
        sys.exit()

############### MAIN RUNTIME SCRIPT ########################
#
# Below is the actual script that is run when this file is run through
# the python interpreter. The code above defines functions used by the below
# runtime script.
#

if __name__ == '__main__':
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
    file_log = None
    
    for file_path in DATA_FILES:
        file_log = open(OUTPUT_FOLDER + '/conversion.log', 'a')
        try:
#        if 1:
            t0 = getTime()
            dpath, dfile = os.path.split(file_path)
            print "Processing file %d / %d. \r" % (file_proc_count + 1, total_file_count)
                
            et_model, sub = getInfoFromPath(file_path)
            print 'tracker: {et_model}, sub: {sub}'.format(et_model=et_model, sub=sub)
            
            et_dir = nabs(r"%s/%s" % (OUTPUT_FOLDER, et_model))
            if not os.path.exists(et_dir):
                os.mkdir(et_dir)
    
            if et_model == 'eyetribe':
                # open eyetribe files in update mode so time stamp issue can
                # be fixed in files.
                hub_file = openHubFile(dpath, dfile, 'a')
            else:
                hub_file = openHubFile(dpath, dfile, 'r')
            
            if not checkFileIntegrity(hub_file):
#                file_log.write('[FILE_CORRUPT]\tfile: {file_path}\n'.format(file_path=file_path ))
                raise ConversionError(str("FILE_CORRUPT"))
            
            screen_measurments, et_model = getScreenMeasurements(file_path, et_model_display_configs)
            wide_format_samples_by_session = convertEDQ(hub_file,
                                                        screen_measurments,
                                                        et_model
                                             )
            if wide_format_samples_by_session == None or len(wide_format_samples_by_session) == 0:
                print "\n>>>>>>\nERROR processing Hdf5 file: %s\n\tFile has " \
                      "no 'FS' BLOCK COND VARS.\n\tSKIPPING FILE.\n<<<<<<\n" % (file_path)
                raise ConversionError(str("NO_BLOCK_DATA"))
    
            wide_format_samples = []
            for output_samples in wide_format_samples_by_session:
                wide_format_samples.extend(output_samples)
    
            scount += len(wide_format_samples)
            
            data_wide = np.array(wide_format_samples, dtype=wide_row_dtype)            
            tracking_eye = parseTrackerMode(data_wide['eyetracker_mode'][0])
            
            if (et_model == 'dpi'):
                ### Handle DPI data START ###
                data_wide, _ = filter_trackloss(filter_bilateral(data_wide), et_model)
            
                if CALIBRATE_DPI:
                    t1 = getTime()
                    print "DPI calibration"
        
                    if (len(np.unique(data_wide['session_id'])) > 1):
                        data_wide, _, exp_session_id = handle_dpi_multisession(data_wide)
                    else:
                        exp_session_id = data_wide['session_id'][0]
                        
                    #Calibration settings
                    poly_type = calibration_settings_set[0]['poly_type']
                    cal_point_set = calibration_settings_set[0]['cal_point_set']
                    min_cal_points = calibration_settings_set[0]['min_cal_points']
                    win_select_func = calibration_settings_set[0]['win_select_func']
                    units = calibration_settings_set[0]['units']

                    args={
                          'win_size': calibration_settings_set[0]['win_size'],
                          'win_type': calibration_settings_set[0]['win_type'],
                          'window_skip': calibration_settings_set[0]['window_skip'],
                          'wsa': [calibration_settings_set[0]['wsa']],
                          'target_count' : 25
                          }
                    
                    cal_points = cal_point_sets[cal_point_set]
                    cal_block = data_wide['BLOCK'] == 'DPICAL'
                    exp_block = data_wide['BLOCK'] == 'FS' #TODO: deal with other blocks
                    DATA_CAL = data_wide[cal_block]
                    
                    if np.sum(cal_block) == 0:
                        print 'DPI calibration block missing'
                        raise ConversionError(str("DPI_CAL_NO_DATA"))
                    if np.sum(exp_block) == 0:
                        print 'DPI experiment block missing'
                        raise ConversionError(str("DPI_EXP_NO_DATA"))
                   
                   ### CALIBRATION START ###
                    data_wide_raw = np.copy(data_wide)
  
                    stim_CAL = win_select_funcs[win_select_func](DATA_CAL, **args)
                    if len(stim_CAL)>25:
                        print 'Multiple calibrations per session...skipping'
                        raise ConversionError(str("DPI_MULTIPLE_CAL_PER_SESS"))
                        
                    cal_ind=stim_CAL['ROW_INDEX'] == cal_points[:, None]
                    cal_ind=np.array(np.sum(cal_ind, axis=0), dtype=bool)
                    
                    ### Handle missing calibration points: replace with random ones
                    stim_key_x = '_'.join((tracking_eye[0], units, 'fix', 'x'))
                    stim_key_y = '_'.join((tracking_eye[0], units, 'fix', 'y'))
                    if (np.isnan(stim_CAL[stim_key_x][cal_ind]).any()) | \
                       (np.isnan(stim_CAL[stim_key_y][cal_ind]).any()):
                        print 'Calibration points missing...Trying to replace'
                        
                        valid_cal_ind = np.bitwise_and(np.isfinite(stim_CAL[stim_key_x]), 
                                                       np.isfinite(stim_CAL[stim_key_y])
                                        )
                        
                        cal_ind = np.bitwise_and(cal_ind, valid_cal_ind)
                        
                        extra_cal_ind = np.bitwise_xor(valid_cal_ind, cal_ind)
                        rand_p_count = cal_point_set-np.sum(cal_ind) #how many points are missing
                        if rand_p_count > np.sum(extra_cal_ind): #if needed points exceed available points
                            rand_p_count = np.sum(extra_cal_ind) #select only available amount of extra points
                        rand_cal_ind = np.random.choice(np.arange(len(extra_cal_ind))[extra_cal_ind], rand_p_count, replace=False)
                        extra_cal_ind[:]=False
                        extra_cal_ind[rand_cal_ind] = True
                        
                        cal_ind = np.bitwise_or(cal_ind, extra_cal_ind)
                        
                        if (np.sum(cal_ind)<min_cal_points):
                            print 'Only %d points available..Skipping'%np.sum(cal_ind)
                            file_log.write('[DPI_CAL_NO_POINTS]\tfile: {file_path}\tAvailable calibration points: {cal_p_available}\n'.format(cal_p_available=np.sum(cal_ind), file_path=file_path ))
                            raise ConversionError(str("DPI_CAL_NO_POINTS"))
                        else:
                            file_log.write('[DPI_CAL_POINTS_REPLACED]\tfile: {file_path}\tReplaced calibration points: {rand_p_count}\n'.format(rand_p_count=rand_p_count, file_path=file_path ))
                        
                    ####
                    #Conversion to degrees
                    session_info_dict = getSessionDataFromMsgEvents(hub_file)
                    session_info = session_info_dict[exp_session_id]
                    display_size_pix = session_info['display_width_pix'], session_info['display_height_pix']    
                    display_size_mm = screen_measurments['screen_width'], screen_measurments['screen_height']                
                    pix2deg = VisualAngleCalc(display_size_mm, display_size_pix,
                                              screen_measurments['eye_distance']).pix2deg
                                              
                    for eye in tracking_eye:               
        
                        Px, Py = build_polynomial(stim_CAL['_'.join((eye, units, 'fix_x'))][cal_ind], 
                                                  stim_CAL['_'.join((eye, units, 'fix_y'))][cal_ind], poly_type)
                        
                        calX, calY = np.linalg.lstsq(Px, stim_CAL[stim_pos_mappings[units]+'x'][cal_ind])[0], \
                                     np.linalg.lstsq(Py, stim_CAL[stim_pos_mappings[units]+'y'][cal_ind])[0]
                        
                        Px_data, Py_data = build_polynomial(data_wide_raw['_'.join((eye, units, 'x'))], 
                                                            data_wide_raw['_'.join((eye, units, 'y'))] , poly_type)
                        
        
                        data_wide['_'.join((eye, units, 'x'))] = np.dot(Px_data, calX)
                        data_wide['_'.join((eye, units, 'y'))] = np.dot(Py_data, calY)
        
                        (data_wide['_'.join((eye, 'angle', 'x'))], 
                         data_wide['_'.join((eye, 'angle', 'y'))])=pix2deg(data_wide['_'.join((eye, units, 'x'))],
                                                                           data_wide['_'.join((eye, units, 'y'))])
                                            
                    ### DPI calibration END ### 
                    
                    #Save only FS block
                    data_wide = data_wide[exp_block]

                    ### Empty recording check 
                    _, loss_count = filter_trackloss(data_wide, et_model)
                    check_eye = dict()
                    #Becomes True if all nans found                
                    check_eye['right'] = loss_count['right'] == len(data_wide['right_gaze_x'])
                    check_eye['left'] = loss_count['left'] == len(data_wide['left_gaze_x'])
                    if check_eye['right'] &  check_eye['left']:
                        print "Recording does not contain any data...skipping"
                        raise ConversionError(str("DPI_NO_DATA")) 
                    else:
                        file_log.write('[DPI_CAL_OK]\tfile: {file_path}\tCalibrated using {cal_p} points\n'.format(cal_p=np.sum(cal_ind), file_path=file_path ))
                        print 'DPI calibration duration: ', getTime()-t1  

                ### Handle DPI data END ###
                
            else:
                ### Handle VOG data START ###
                data_wide, loss_count = filter_trackloss(data_wide, et_model)
                
                #All targets check                        
                if len(np.unique(data_wide['ROW_INDEX']))<25: #continue, if at least half of the targets recorded
                    print "Not enough data recorded...skipping"
                    raise ConversionError(str("VOG_NO_ENOUGH_STIM"))  
                if len(np.unique(data_wide['ROW_INDEX']))!=49: 
                    print "Not all targets recorded"
                    file_log.write('[VOG_NO_ALL_STIM]\tfile: {file_path}\n'.format(file_path=file_path ))
                
                ### Empty recording check 
                check_eye = dict()
                #Becomes True if all nans found                
                check_eye['right'] = loss_count['right'] == len(data_wide['right_gaze_x'])
                check_eye['left'] = loss_count['left'] == len(data_wide['left_gaze_x'])
                if check_eye['right'] &  check_eye['left']:
                    print "Recording does not contain any data...skipping"
                    raise ConversionError(str("VOG_NO_DATA"))                
                
                ### Data integrity checks
                if data_wide['eyetracker_mode'][0] == 'Binocular':
                    # Tracking mode check 
                    if et_model in MONOCULAR_TRACKERS:
                        print 'Wrong tracking mode selected for monocular tracker'
                        raise ConversionError(str("VOG_MODE_SELECT_ERROR")) 
                    
                    # Binocular averaging check
                    if CHECK_BDA & nan_equal(data_wide['left_gaze_x'], data_wide['right_gaze_x']):
                        print 'Binocular data averaged..skipping' 
                        raise ConversionError(str("VOG_BDA")) 
                
                #Monocular eye select fix    
                elif check_eye[tracking_eye[0]]:
                    print "Eye selection error...correcting"
                    eye_corr = check_eye.keys()[check_eye.values().index(False)].title()
                    data_wide['eyetracker_mode'] = eye_corr+' eye' 
                        
                    file_log.write('[VOG_EYE_SELECT_CORRECTION]\tfile: {file_path}\n'.format(file_path=file_path ))
                ###
                
                ### Deal with multisession recordings
                if (len(np.unique(data_wide['session_id'])) > 1):
                    print 'Multiple sessions found'
                    tr_loss = []
                    session_ids = np.unique(data_wide['session_id'])
                    
                    ### Manual multisession handler                    
                    mask_smc = ((special_multisession_cases['subject_id'] == sub )
                              & (special_multisession_cases['eyetracker_model'] == et_model)
                    )
                    
                    if mask_smc.any():
                        session_ids = special_multisession_cases['session_id'][mask_smc]
                        print 'Multisession special case'
                        file_log.write('[VOG_MULTISESSION_SPEC]\tfile: {file_path}\tsid: {sid}\n'.format(file_path=file_path, sid=session_ids[0]))
                    
                    ###

                    for sid in session_ids:
                        mask = data_wide['session_id']==sid
                        if len(np.unique(data_wide['ROW_INDEX'][mask]))==49:
                            _, loss_count = filter_trackloss(data_wide[mask], et_model)
                            tr_loss.append((sid, loss_count['avg'], np.sum(mask)))
                            
                            ### PLot multisession data
                            if plot_multisession:
                                tr_loss_caption = ''
                                for eye in parseTrackerMode(data_wide['eyetracker_mode'][0]):
                                    tr_loss_caption += '{eye} eye: {tr_loss:.2f} %; '.format(eye=eye, tr_loss=100*float(loss_count[eye])/np.sum(mask))
                                
                                _title = 'Tracker: {et_model}, Operator: {operator}, trackloss: {loss}'.format(et_model=et_model,
                                                                                                               operator=data_wide['operator'][0], 
                                                                                                               loss = tr_loss_caption
                                                                                                        )
                                _fname = '{output_dir}/{et_model}/multisession_{et_model}_sub_{sub}_sid_{sid}.png'.format(output_dir=OUTPUT_FOLDER, 
                                                                                                                          et_model=et_model,
                                                                                                                          sub=sub,
                                                                                                                          sid=sid
                                                                                                                   )
                                plot_data(data_wide, title=_title, fname=_fname, ylim=[-30, 30]) 
                            ###
                                
                    tr_loss = np.array(tr_loss)
                    if tr_loss.any():
                        least_loss_ind = np.argmin(np.float32(tr_loss[:,1])/tr_loss[:,2])
                        
                        #Check for 100% trackloss
                        if tr_loss[least_loss_ind][1] < tr_loss[least_loss_ind][2]:
                            sid = tr_loss[least_loss_ind,0]
                            mask = data_wide['session_id']==sid
                            data_wide = data_wide[mask]
                            
                            file_log.write('[VOG_MULTISESSION]\tfile: {file_path}\tsid: {sid}\n'.format(file_path=file_path, sid=sid))
                        else:
                            print "Session does not contain any data...skipping"
                            raise ConversionError(str("VOG_NO_DATA_SESSION"))
                    else:
                        print "Session does not contain any data...skipping"
                        file_log.write('[VOG_MULTISESSION]\tfile: {file_path}\tsid: {sid}\n'.format(file_path=file_path, sid=np.nan))
                        raise ConversionError(str("VOG_NO_DATA_STIM"))
                        
                ### Handle VOG data END ###
                        
            print 'Conversion duration: ', getTime()-t0
    
            ### Save data
            if SAVE_NPY:
                np_file_name = r"%s/%s_%s.npy" % (et_dir, et_model, dfile[:-5]) 
                t0 = getTime()
                np.save(np_file_name, data_wide)
                print 'RAW_NPY save duration: ', getTime()-t0
            
            if SAVE_TXT:
                txt_file_name = r"%s/%s_%s.txt" % (et_dir, et_model, dfile[:-5])  
                t0 = getTime()
                save_as_txt(txt_file_name, data_wide)
                print 'RAW_TXT save duration: ', getTime()-t0
            ###
                
            file_log.write('[CONVERSION_OK]\tfile: {file_path}\n'.format(file_path=file_path ))
            
            ### Plot data
            if et_model in PLOT_TRACKERS:
                _, loss_count = filter_trackloss(data_wide, et_model)
                tr_loss_caption = ''
                for eye in parseTrackerMode(data_wide['eyetracker_mode'][0]):
                    tr_loss_caption += '{eye} eye: {tr_loss:.2f} %; '.format(eye=eye, tr_loss=100*float(loss_count[eye])/len(data_wide))
                                
                _title = 'Tracker: {et_model}, Operator: {operator}, trackloss: {loss}'.format(et_model=et_model,
                                                                                               operator=data_wide['operator'][0], 
                                                                                               loss = tr_loss_caption
                                                                                        )                
                _fname = '{output_dir}/{et_model}/{et_model}_sub_{sub}.png'.format(output_dir=OUTPUT_FOLDER,
                                                                                   et_model=et_model,
                                                                                   sub=sub
                                                                            )
                if et_model == 'dpi':
                    plot_data(data_wide, title=_title, fname=_fname, ylim=[-10, 10])
                else:
                    plot_data(data_wide, title=_title, fname=_fname, ylim=[-30, 30])
            ###
        
        except ConversionError, e:
            print 'Conversion error...skipping'
            file_log.write('[{msg}]\tfile: {file_path}\n'.format(msg=e.message, file_path=file_path ))
            file_log.write('[CONVERSION_ERROR]\tfile: {file_path}\n'.format(file_path=file_path ))
        except:
            print 'Unhandled conversion error...skipping'
            file_log.write('[UNHANDLED_CONVERSION_ERROR]\tfile: {file_path}\n'.format(file_path=file_path ))
        finally:
            file_proc_count += 1
            print 
            if hub_file:
                hub_file.close()
                hub_file = None
            if file_log:
                file_log.close()
        
    end_time = getTime()
  
    print
    print 'Processed File Count:', file_proc_count
    print 'Total Samples Selected for Output:', scount
    print "Total Run Time:", (end_time - start_time)
    print "Samples / Second:", scount / (end_time - start_time)
    
    if file_log:
        file_log.close()

sys.exit()
    