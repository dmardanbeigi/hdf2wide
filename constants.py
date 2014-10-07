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