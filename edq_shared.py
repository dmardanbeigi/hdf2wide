# -*- coding: utf-8 -*-
import os

import numpy as np
import numpy.ma as ma

from constants import *

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

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)        

def add_field(a, descr):
    """Return a new array that is like "a", but has additional fields.

    Arguments:
      a     -- a structured numpy array
      descr -- a numpy type description of the new fields

    The contents of "a" are copied over to the appropriate fields in
    the new array, whereas the new fields are uninitialized.  The
    arguments are not modified.

    >>> sa = numpy.array([(1, 'Foo'), (2, 'Bar')], \
                         dtype=[('id', int), ('name', 'S3')])
    >>> sa.dtype.descr == numpy.dtype([('id', int), ('name', 'S3')])
    True
    >>> sb = add_field(sa, [('score', float)])
    >>> sb.dtype.descr == numpy.dtype([('id', int), ('name', 'S3'), \
                                       ('score', float)])
    True
    >>> numpy.all(sa['id'] == sb['id'])
    True
    >>> numpy.all(sa['name'] == sb['name'])
    True
    """
    if a.dtype.fields is None:
        raise ValueError, "`A' must be a structured numpy array"
    b = np.zeros(a.shape, dtype=a.dtype.descr + descr)
    for name in a.dtype.names:
        b[name] = a[name]
    return b
    
    
### VisualAngleCalc
"""
Copied code from iohub
(trying to keep dependencies low for this conversion script)

__author__ = 'Sol'
"""
arctan = np.arctan2
rad2deg = np.rad2deg
hypot = np.hypot
np_abs = np.abs
np_sqrt = np.sqrt

class VisualAngleCalc(object):
    def __init__(self, display_size_mm, display_res_pix, eye_distance_mm=None):
        """
        Used to store calibrated surface information and eye to screen distance
        so that pixel positions can be converted to visual degree positions.

        Note: The information for display_size_mm,display_res_pix, and default
        eye_distance_mm could all be read automatically when opening a ioDataStore
        file. This automation should be implemented in a future release.
        """
        self._display_width = display_size_mm[0]
        self._display_height = display_size_mm[1]
        self._display_x_resolution = display_res_pix[0]
        self._display_y_resolution = display_res_pix[1]
        self._eye_distance_mm = eye_distance_mm
        self.mmpp_x = self._display_width / self._display_x_resolution
        self.mmpp_y = self._display_height / self._display_y_resolution

    def pix2deg(self, pixel_x, pixel_y=None, eye_distance_mm=None):
        """
        Stimulus positions (pixel_x,pixel_y) are defined in x and y pixel units,
        with the origin (0,0) being at the **center** of the display, as to match
        the PsychoPy pix unit coord type.

        The pix2deg method is vectorized, meaning that is will perform the
        pixel to angle calculations on all elements of the provided pixel
        position numpy arrays in one numpy call.

        The conversion process can use either a fixed eye to calibration
        plane distance, or a numpy array of eye distances passed as
        eye_distance_mm. In this case the eye distance array must be the same
        length as pixel_x, pixel_y arrays.
        """
        eye_dist_mm = self._eye_distance_mm
        if eye_distance_mm is not None:
            eye_dist_mm = eye_distance_mm

        x_mm = self.mmpp_x * pixel_x
        y_mm = self.mmpp_y * pixel_y

        Ah = arctan(x_mm, hypot(eye_dist_mm, y_mm))
        Av = arctan(y_mm, hypot(eye_dist_mm, x_mm))

        return rad2deg(Ah), rad2deg(Av)
###
        
        
def initStim(data):
    """
    Creates /one-row-per-target/ data matrix 
    
    @author: Raimondas Zemblys
    @email: raimondas.zemblys@humlab.lu.se
    """
    stim_change_ind = np.where(np.hstack((1,np.diff(data['trial_id']))) == 1)
    stim_change_count = len(stim_change_ind[0])
    
    wsa_list=np.empty_like(data['eyetracker_model'][stim_change_ind])
    wsa_list[:]='nan'
    
    stim = np.array(
               zip(
                   data['eyetracker_model'][stim_change_ind],
                   data['eyetracker_sampling_rate'][stim_change_ind],                 
                   data['eyetracker_mode'][stim_change_ind],
                   np.ones(stim_change_count) * getGeometry(data[0]), 
                   data['operator'][stim_change_ind],

                   data['subject_id'][stim_change_ind],
                   data['trial_id'][stim_change_ind], 
                   data['ROW_INDEX'][stim_change_ind], 
                   data['dt'][stim_change_ind], 
                   data['TRIAL_START'][stim_change_ind],
                   data['TRIAL_END'][stim_change_ind],
                   data['posx'][stim_change_ind],
                   data['posy'][stim_change_ind],
                   data['target_angle_x'][stim_change_ind],
                   data['target_angle_y'][stim_change_ind],
                   
                   wsa_list.tolist(),
                   *np.zeros((73, stim_change_count)) * np.nan
               ), dtype=stim_dtype
           )
    
    #Remove first and last stimuli
    return stim[1:-1]
    
    
def detect_rollingWin(data, **args):
    """
    Fills /one-row-per-target/ data matrix
    
    @author: Raimondas Zemblys
    @email: raimondas.zemblys@humlab.lu.se
    """    
    win_size = args['win_size']
    win_type = args['win_type']
    win_size_sample = np.int16(win_size*data['eyetracker_sampling_rate'][0])+1
    window_skip = args['window_skip']
    selection_algorithms = args['wsa']    
    
    measures=dict()
    stim_full=[]
    for eye in parseTrackerMode(data['eyetracker_mode'][0]):
        if win_type ==  'sample':     
            measures.update(rolling_measures_sample(data, eye, 'gaze', win_size_sample))
            measures.update(rolling_measures_sample(data, eye, 'angle', win_size_sample))
        if win_type ==  'time':     
            measures.update(rolling_measures_time(data, eye, 'gaze', win_size))
            measures.update(rolling_measures_time(data, eye, 'angle', win_size)) 
        
    for wsa in selection_algorithms:
        stim = initStim(data)
        stim['wsa'][:] = wsa
        stim['win_size'][:] =  win_size
        stim['window_skip'][:] =  window_skip
        
        for eye in parseTrackerMode(data['eyetracker_mode'][0]):
            for stim_ind, stim_row in enumerate(stim):
                
                analysis_range = (data['time'] >= stim_row['TRIAL_START']+window_skip) \
                               & (data['time'] <= stim_row['TRIAL_START']+1-win_size)
                #needs to be a number to identify starting sample of a window               
                analysis_range=np.squeeze(np.argwhere(analysis_range==True))
                
                analysis_range_full= (data['time'] >= stim_row['TRIAL_START']) \
                                   & (data['time'] <= stim_row['TRIAL_START']+1)
                stim['total_sample_count'][stim_ind] = np.sum(analysis_range_full)
                #invalid_sample_count should be the same for gaze and angle
                stim['_'.join((eye, 'invalid_sample_count'))][stim_ind] = np.sum(np.isnan(data[eye+'_gaze_x'][analysis_range_full]) |
                                                                                 np.isnan(data[eye+'_gaze_y'][analysis_range_full])
                                                                          )
                
                for units in ['gaze', 'angle']:
                    measures_rangeRMS = measures['_'.join((eye, units, 'RMS'))][analysis_range]
                    measures_rangeACC = measures['_'.join((eye, units, 'ACC'))][analysis_range]
                    measures_rangeSTD = measures['_'.join((eye, units, 'STD'))][analysis_range]
                    
                    if wsa == 'fiona':
                        measures_range = measures_rangeRMS #Fiona RMS 
                    elif wsa == 'dixon1':
                        measures_range = measures_rangeACC*measures_rangeSTD #Dixons's measure No. 1
                    elif wsa == 'dixon2':
                        measures_range = measures_rangeACC+measures_rangeSTD #Dixons's measure No. 2
                    elif wsa == 'dixon3':
                        measures_range = measures_rangeACC**2+measures_rangeSTD**2 #Dixons's measure No. 3
                    elif wsa == 'jeff':
                        #Jeff's measure                        
                        std_thres = np.nanmin(measures_rangeSTD)*5
                        measures_rangeACC[measures_rangeSTD>std_thres] = np.nan
                        measures_range = measures_rangeACC

                    if np.sum(np.isfinite(measures_range)) > 0: #handle all-nan slice
                        IND = analysis_range[np.nanargmin(measures_range)]
                        
                        #save measures to stim
                        stim['_'.join((eye, units, 'ind'))][stim_ind] = IND
                        stim['_'.join((eye, units, 'window_onset'))][stim_ind] = data['time'][IND]-stim_row['TRIAL_START']

                        for key in measures.keys():
                            stim[key][stim_ind]=measures[key][IND]  
            
        stim_full.extend(stim.tolist())
                  
    return np.array(stim_full, dtype=stim_dtype)
    
    
def rolling_measures_sample(data, eye, units, win_size_sample):
    """
    Calculates rolling window measures 
    
    @author: Raimondas Zemblys
    @email: raimondas.zemblys@humlab.lu.se
    """  
    measures=dict()
    
    #RMS    
    isd = np.diff([data['_'.join((eye, units, 'x'))], 
                   data['_'.join((eye, units, 'y'))]], axis=1).T

    measures['_'.join((eye, units, 'RMS', 'x'))] = np.sqrt(np.mean(np.square(rolling_window(isd[:,0], win_size_sample-1)), 1))
    measures['_'.join((eye, units, 'RMS', 'y'))] = np.sqrt(np.mean(np.square(rolling_window(isd[:,1], win_size_sample-1)), 1))    
    measures['_'.join((eye, units, 'RMS'))] = np.hypot(measures['_'.join((eye, units, 'RMS', 'x'))],
                                                       measures['_'.join((eye, units, 'RMS', 'y'))]   
                                              )
    ###
                                              
    rolling_data_x = rolling_window(data['_'.join((eye, units, 'x'))], win_size_sample)
    rolling_data_y = rolling_window(data['_'.join((eye, units, 'y'))], win_size_sample)
    rolling_time = rolling_window(data['time'], win_size_sample)
    
    measures['_'.join((eye, units, 'sample_count'))] = np.ones(len(rolling_data_x)) * win_size_sample
    measures['_'.join((eye, units, 'actual_win_size'))] = rolling_time[:,-1]-rolling_time[:,0]
    
    #STD  
    measures['_'.join((eye, units, 'STD', 'x'))] = np.std(rolling_data_x, axis=1)
    measures['_'.join((eye, units, 'STD', 'y'))] = np.std(rolling_data_y, axis=1)
    measures['_'.join((eye, units, 'STD'))] = np.hypot(measures['_'.join((eye, units, 'STD', 'x'))],
                                                       measures['_'.join((eye, units, 'STD', 'y'))]   
                                              )
    
    #ACC
    fix = [np.median(rolling_data_x, axis=1),
           np.median(rolling_data_y, axis=1)]

    measures['_'.join((eye, units, 'ACC', 'x'))] = fix[0] - data[stim_pos_mappings[units]+'x'][:-win_size_sample+1]
    measures['_'.join((eye, units, 'ACC', 'y'))] = fix[1] - data[stim_pos_mappings[units]+'y'][:-win_size_sample+1]
    measures['_'.join((eye, units, 'ACC'))] = np.hypot(measures['_'.join((eye, units, 'ACC', 'x'))],
                                                       measures['_'.join((eye, units, 'ACC', 'y'))]   
                                              )
    
    #Fix
    measures['_'.join((eye, units, 'fix', 'x'))] = fix[0]
    measures['_'.join((eye, units, 'fix', 'y'))] = fix[1]
    
    #Other measures
    measures['_'.join((eye, units, 'range', 'x'))] = np.max(rolling_data_x, axis=1)-np.min(rolling_data_x, axis=1)
    measures['_'.join((eye, units, 'range', 'y'))] = np.max(rolling_data_y, axis=1)-np.min(rolling_data_y, axis=1)
    
    return measures

def rolling_measures_time(data, eye, units, win_size):
    """
    Calculates rolling window measures 
    
    @author: Raimondas Zemblys
    @email: raimondas.zemblys@humlab.lu.se
    """  
    measures=dict()
    fs = data['eyetracker_sampling_rate'][0]
    win_size_sample = np.int16(win_size*fs)+1
    
    #adjust window size to account for the uncertainty of a measurement    
    win_size+=1.0/fs/2
    
    #get masks for windows based on time    
    temp_acc=np.diff(data['time'])
    rolling_temp_acc_for_data=np.cumsum(np.insert(rolling_window(temp_acc,win_size_sample*2-1), 0, np.zeros(len(temp_acc)-(win_size_sample-1)*2), axis=1), axis=1)
    rolling_temp_acc_for_isd=np.cumsum(rolling_window(temp_acc,win_size_sample*2-1), axis=1)
    mask_data=ma.getmaskarray(ma.masked_greater(rolling_temp_acc_for_data, win_size))
    mask_isd=ma.getmaskarray(ma.masked_greater(rolling_temp_acc_for_isd, win_size))
  
    ### RMS
    isd = np.diff([data['_'.join((eye, units, 'x'))], 
                   data['_'.join((eye, units, 'y'))]], axis=1).T
    
    rolling_isd_x = ma.array(rolling_window(isd[:,0], win_size_sample*2-1),mask=mask_isd) 
    rolling_isd_y = ma.array(rolling_window(isd[:,1], win_size_sample*2-1),mask=mask_isd)

    RMS=[]
    for rms in [np.sqrt(np.mean(np.square(rolling_isd_x), 1)),
                np.sqrt(np.mean(np.square(rolling_isd_y), 1)),
                ]:
        rms_tmp = ma.getdata(rms)
        mask = ma.getmask(rms)
        rms_tmp[mask]=np.nan
        RMS.append(rms_tmp)
    
    measures['_'.join((eye, units, 'RMS', 'x'))] = RMS[0]
    measures['_'.join((eye, units, 'RMS', 'y'))] = RMS[1]    
    measures['_'.join((eye, units, 'RMS'))] = np.hypot(RMS[0], RMS[1])
    ###
    
    rolling_data_x = ma.array(rolling_window(data['_'.join((eye, units, 'x'))], win_size_sample*2),mask=mask_data) 
    rolling_data_y = ma.array(rolling_window(data['_'.join((eye, units, 'y'))], win_size_sample*2),mask=mask_data) 
    rolling_time = ma.array(rolling_window(data['time'], win_size_sample*2),mask=mask_data)
    
    measures['_'.join((eye, units, 'sample_count'))] = np.sum(mask_data, axis=1)
    
    notmasked_edges=ma.notmasked_edges(rolling_time, axis=1)
    start_times = ma.getdata(rolling_time[notmasked_edges[0][0],notmasked_edges[0][1]])
    end_times = ma.getdata(rolling_time[notmasked_edges[1][0],notmasked_edges[1][1]])
    measures['_'.join((eye, units, 'actual_win_size'))] = end_times-start_times
    return measures
    
def getGeometry(data):
    """
    Calculates pix2deg values, based on simple geometry 
    
    @author: Raimondas Zemblys
    @email: raimondas.zemblys@humlab.lu.se
    """    
    return np.mean((1/(np.degrees(2*np.arctan(data['screen_width']/(2*data['eye_distance'])))/data['display_width_pix']),
                    1/(np.degrees(2*np.arctan(data['screen_height']/(2*data['eye_distance'])))/data['display_height_pix'])))
                    
def filter_trackloss(data_wide, et_model=None, fill=np.nan):
    """
    Trackloss filter. Replaces invalid samples with /fill/
    
    @author: Raimondas Zemblys
    @email: raimondas.zemblys@humlab.lu.se
    """   
    #TODO: Filter off-screen, off-pshysical limit samples
    data = np.copy(data_wide) #remove if memory issues
    for eye in ['left', 'right']:
        trackloss = (data['_'.join((eye, 'gaze_x'))] == et_nan_values[et_model]['x']) | \
                    (data['_'.join((eye, 'gaze_y'))] == et_nan_values[et_model]['y'])
        if et_model == 'dpi':
            trackloss = np.bitwise_or(trackloss, data['status'] < 4.0)
            
        data['_'.join((eye, 'gaze_x'))][trackloss] = fill
        data['_'.join((eye, 'gaze_y'))][trackloss] = fill
        data['_'.join((eye, 'angle_x'))][trackloss] = fill
        data['_'.join((eye, 'angle_y'))][trackloss] = fill
        
    return data, np.sum(trackloss)