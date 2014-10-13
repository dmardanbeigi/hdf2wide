__author__ = 'Sol'
__version__ = 'RZ'
import numpy as np

et_nan_values = dict()
et_nan_values['eyefollower'] = {'x': 0.0, 'y': 0.0}
et_nan_values['eyelink'] =  {'x': 0.0, 'y': 0.0}
et_nan_values['eyetribe'] = {'x': -840.0, 'y': 525.0}
et_nan_values['hispeed1250'] =  {'x': 0.0, 'y': 0.0}
et_nan_values['hispeed240'] =  {'x': 0.0, 'y': 0.0}
et_nan_values['red250'] =  {'x': 0.0, 'y': 0.0}
et_nan_values['red500'] =  {'x': 0.0, 'y': 0.0}
et_nan_values['redm'] =  {'x': 0.0, 'y': 0.0}
et_nan_values['t60xl'] = {'x': -1.0, 'y': -1.0}
et_nan_values['tx300'] = {'x': -1.0, 'y': -1.0}
et_nan_values['x2'] = {'x': -1.0, 'y': -1.0}
et_nan_values['dpi'] = {'x': -10000, 'y': -10000}

wide_row_dtype = np.dtype([
    ('subject_id', np.uint8),
    ('display_refresh_rate', np.float32),
    ('eyetracker_model', str, 32),
    ('dot_deg_sz', np.float32),
    ('eyetracker_sampling_rate', np.float32),
    ('eyetracker_mode', str, 16),
    ('fix_stim_center_size_pix', np.uint8),
    ('operator', str, 8),
    ('eyetracker_id', np.uint8),
    ('display_width_pix', np.float32),
    ('display_height_pix', np.float32),
    ('screen_width', np.float32),
    ('screen_height', np.float32),
    ('eye_distance', np.float32),
    ('SESSION_ID', np.uint8),
    ('trial_id', np.uint16),
    ('TRIAL_START', np.float32),
    ('TRIAL_END', np.float32),
    ('posx', np.float32),
    ('posy', np.float32),
    ('dt', np.float32),
    ('ROW_INDEX', np.uint8),
    ('BLOCK', str, 6),
    ('session_id', np.uint8),
    ('device_time', np.float32),
    ('time', np.float32),
    ('left_gaze_x', np.float32),
    ('left_gaze_y', np.float32),
    ('left_pupil_measure1', np.float32),
    ('right_gaze_x', np.float32),
    ('right_gaze_y', np.float32),
    ('right_pupil_measure1', np.float32),
    ('status', np.uint8),
    ('target_angle_x', np.float32),
    ('target_angle_y', np.float32),
    ('left_angle_x', np.float32),
    ('left_angle_y', np.float32),
    ('right_angle_x', np.float32),
    ('right_angle_y', np.float32)
    ])
    


from collections import OrderedDict
msg_txt_mappings = OrderedDict()
msg_txt_mappings['Participant ID'] = 'subject_id'
msg_txt_mappings['Monitor refresh rate (Hz)'] = 'display_refresh_rate'
msg_txt_mappings['Eye tracker'] = 'eyetracker_model'
msg_txt_mappings['dotStimSize (deg)'] = 'dot_deg_sz'
msg_txt_mappings['Tracker SamplingRate'] = 'eyetracker_sampling_rate'
msg_txt_mappings['Tracker mode'] = 'eyetracker_mode'
msg_txt_mappings['dotStimCenter (px)'] = 'fix_stim_center_size_pix'
msg_txt_mappings['Operator'] = 'operator'
msg_txt_mappings['Tracker ID'] = 'eyetracker_id'
msg_txt_mappings['Stimulus Screen ID'] = 'display_width_pix'
msg_txt_mappings['Stimulus Screen ID2'] = 'display_height_pix'
msg_txt_mapping_keys = msg_txt_mappings.keys()
msg_txt_mapping_values = msg_txt_mappings.values()

#Fixes ROW_INDEX in DPI calibration routine for 2014 Apr-May EDQ recordings
dpi_cal_fix = dict([
    (0,21),
    (1,7),
    (2,22),
    (3,14),
    (4,20),
    (5,10),
    (6,17),
    (7,6),
    (8,15),
    (9,0),
    (10,18),
    (11,11),
    (12,4),
    (13,5),
    (14,13),
    (15,3),
    (16,12),
    (17,19),
    (18,24),
    (19,8),
    (20,2),
    (21,16),
    (22,1),
    (23,23),
    (24,9),
    ])

stim_dtype = np.dtype([

    ('eyetracker_model', str, 32),
    ('eyetracker_sampling_rate', np.float32),
    ('eyetracker_mode', str, 16),
    ('px2deg', np.float32), 
    ('operator', str, 8),

    ('subject_id', np.uint8),
    ('trial_id', np.uint16),
    ('ROW_INDEX', np.uint8),
    ('dt', np.float32),
    ('TRIAL_START', np.float32),
    ('TRIAL_END', np.float32),
    ('posx', np.float32),
    ('posy', np.float32),
    ('target_angle_x', np.float32),
    ('target_angle_y', np.float32),
    
    ('wsa', str, 32),
    ('win_size', np.float32),
    ('window_skip', np.float32),
    
    ('invalid_sample_count', np.float32),

    ('left_gaze_ind', np.float32),
    ('left_window_onset', np.float32),
    ('left_sample_count', np.float32),
    ('left_actual_win_size', np.float32),
    ('left_gaze_ACC', np.float32),
    ('left_gaze_ACC_x', np.float32),
    ('left_gaze_ACC_y', np.float32),
    ('left_gaze_RMS', np.float32),
    ('left_gaze_RMS_x', np.float32),
    ('left_gaze_RMS_y', np.float32),
    ('left_gaze_STD', np.float32),
    ('left_gaze_STD_x', np.float32),
    ('left_gaze_STD_y', np.float32),
    ('left_gaze_fix_x', np.float32),
    ('left_gaze_fix_y', np.float32),
    ('left_gaze_range_x', np.float32),
    ('left_gaze_range_y', np.float32),
    ('left_angle_ind', np.float32),
    ('left_angle_ACC', np.float32),
    ('left_angle_ACC_x', np.float32),
    ('left_angle_ACC_y', np.float32),
    ('left_angle_RMS', np.float32),
    ('left_angle_RMS_x', np.float32),
    ('left_angle_RMS_y', np.float32),
    ('left_angle_STD', np.float32),
    ('left_angle_STD_x', np.float32),
    ('left_angle_STD_y', np.float32),
    ('left_angle_fix_x', np.float32),
    ('left_angle_fix_y', np.float32),
    ('left_angle_range_x', np.float32),
    ('left_angle_range_y', np.float32),
    

    ('right_gaze_ind', np.float32),
    ('right_window_onset', np.float32),
    ('right_sample_count', np.float32),
    ('right_actual_win_size', np.float32),
    ('right_gaze_ACC', np.float32),
    ('right_gaze_ACC_x', np.float32),
    ('right_gaze_ACC_y', np.float32),
    ('right_gaze_RMS', np.float32),
    ('right_gaze_RMS_x', np.float32),
    ('right_gaze_RMS_y', np.float32),
    ('right_gaze_STD', np.float32),
    ('right_gaze_STD_x', np.float32),
    ('right_gaze_STD_y', np.float32),
    ('right_gaze_fix_x', np.float32),
    ('right_gaze_fix_y', np.float32),
    ('right_gaze_range_x', np.float32),
    ('right_gaze_range_y', np.float32),
    ('right_angle_ind', np.float32),
    ('right_angle_ACC', np.float32),
    ('right_angle_ACC_x', np.float32),
    ('right_angle_ACC_y', np.float32),
    ('right_angle_RMS', np.float32),
    ('right_angle_RMS_x', np.float32),
    ('right_angle_RMS_y', np.float32),
    ('right_angle_STD', np.float32),
    ('right_angle_STD_x', np.float32),
    ('right_angle_STD_y', np.float32),
    ('right_angle_fix_x', np.float32),
    ('right_angle_fix_y', np.float32),
    ('right_angle_range_x', np.float32),
    ('right_angle_range_y', np.float32),
    ])
    
stim_pos_mappings=dict([
    ('angle','target_angle_'),
    ('gaze', 'pos'),
    ])
# iohub EventConstants values, as of June 13th, 2014.
# Copied so that iohub does not need to be a dependency of conversion script
KEYBOARD_INPUT = 20
KEYBOARD_KEY = 21
KEYBOARD_PRESS = 22
KEYBOARD_RELEASE = 23

MOUSE_INPUT = 30
MOUSE_BUTTON = 31
MOUSE_BUTTON_PRESS = 32
MOUSE_BUTTON_RELEASE = 33
MOUSE_DOUBLE_CLICK = 34
MOUSE_SCROLL = 35
MOUSE_MOVE = 36
MOUSE_DRAG = 37

TOUCH = 40
TOUCH_MOVE = 41
TOUCH_PRESS = 42
TOUCH_RELEASE = 43

EYETRACKER = 50
MONOCULAR_EYE_SAMPLE = 51
BINOCULAR_EYE_SAMPLE = 52
FIXATION_START = 53
FIXATION_END = 54
SACCADE_START = 55
SACCADE_END = 56
BLINK_START = 57
BLINK_END = 58

GAMEPAD_STATE_CHANGE = 81
GAMEPAD_DISCONNECT = 82

DIGITAL_INPUT = 101
ANALOG_INPUT = 102
THRESHOLD = 103

SERIAL_INPUT = 105
SERIAL_BYTE_CHANGE = 106

MULTI_CHANNEL_ANALOG_INPUT = 122

MESSAGE = 151
LOG = 152