# -*- coding: utf-8 -*-
import os

import numpy as np
import numpy.ma as ma
import matplotlib.pylab as plt
plt.ion()

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
        'nothing to commit (working directory clean)') == -1 and \
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
        
        
def initStim(data, n):
    """
    Creates /one-row-per-target/ data matrix 
    
    @author: Raimondas Zemblys
    @email: raimondas.zemblys@humlab.lu.se
    """
    #Independent variables
    INCLUDE_IVs = ['eyetracker_model',
                   'eyetracker_sampling_rate',
                   'eyetracker_mode',
                   'operator',
                   'subject_id',
                   'exp_date',
                   'trial_id',
                   'ROW_INDEX',
                   'dt',
                   'TRIAL_START',
                   'TRIAL_END',
                   'posx',
                   'posy',
                   'target_angle_x',
                   'target_angle_y',
                   ]  
        
    stim_change_ind = np.where(np.hstack((1,np.diff(data['trial_id']))) == 1)
    stim_change_count = len(stim_change_ind[0])    
    
    stim = np.array( np.ones(stim_change_count)*np.nan, dtype=stim_dtype)
    
    for stim_key_IV in INCLUDE_IVs:
        stim[stim_key_IV][:] = data[stim_key_IV][stim_change_ind]
    
    stim['px2deg'] = getGeometry(data[0])
    
    #Remove first and last stimuli
    ind = (stim['trial_id']==0) | (stim['trial_id']==n+1)
    return stim[~ind]
    
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
    target_count = args['target_count']
    
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
        stim = initStim(data, target_count)
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
                        if np.size(measures_range) == 1:#handle one sample slice
                            measures_range = np.array([measures_range])
                            analysis_range = np.array([analysis_range])
                            
                        IND = analysis_range[np.nanargmin(measures_range)]

                        #save measures to stim
                        stim['_'.join((eye, units, 'ind'))][stim_ind] = IND
                        stim['_'.join((eye, units, 'window_onset'))][stim_ind] = data['time'][IND]-stim_row['TRIAL_START']
                        
                        for key in filter(lambda x: '_'.join((eye, units)) in x, measures.keys()):
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
    
    #Data
    rolling_data_x = rolling_window(data['_'.join((eye, units, 'x'))], win_size_sample)
    rolling_data_y = rolling_window(data['_'.join((eye, units, 'y'))], win_size_sample)
    
    #Position error
    err_x = data['_'.join((eye, units, 'x'))] - data[stim_pos_mappings[units]+'x']
    err_y = data['_'.join((eye, units, 'y'))] - data[stim_pos_mappings[units]+'y']
    
    rolling_err_x = rolling_window(err_x, win_size_sample)
    rolling_err_y = rolling_window(err_y, win_size_sample)
    
    #Time
    rolling_time = rolling_window(data['time'], win_size_sample)
    measures['_'.join((eye, units, 'sample_count'))] = np.ones(len(rolling_time)) * win_size_sample
    measures['_'.join((eye, units, 'actual_win_size'))] = rolling_time[:,-1]-rolling_time[:,0]
    
    #RMS    
    isd = np.diff([data['_'.join((eye, units, 'x'))], 
                   data['_'.join((eye, units, 'y'))]], axis=1).T

    measures['_'.join((eye, units, 'RMS', 'x'))] = np.sqrt(np.mean(np.square(rolling_window(isd[:,0], win_size_sample-1)), 1))
    measures['_'.join((eye, units, 'RMS', 'y'))] = np.sqrt(np.mean(np.square(rolling_window(isd[:,1], win_size_sample-1)), 1))    
    measures['_'.join((eye, units, 'RMS'))] = np.hypot(measures['_'.join((eye, units, 'RMS', 'x'))],
                                                       measures['_'.join((eye, units, 'RMS', 'y'))]   
                                              )
    #RMS of PE
    isd = np.diff([err_x, err_y], axis=1).T
    measures['_'.join((eye, units, 'RMS_PE', 'x'))] = np.sqrt(np.mean(np.square(rolling_window(isd[:,0], win_size_sample-1)), 1))
    measures['_'.join((eye, units, 'RMS_PE', 'y'))] = np.sqrt(np.mean(np.square(rolling_window(isd[:,1], win_size_sample-1)), 1))    
    measures['_'.join((eye, units, 'RMS_PE'))] = np.hypot(measures['_'.join((eye, units, 'RMS_PE', 'x'))],
                                                          measures['_'.join((eye, units, 'RMS_PE', 'y'))]   
                                                 )
    ###
                                                
    ### STD  
    measures['_'.join((eye, units, 'STD', 'x'))] = np.std(rolling_data_x, axis=1)
    measures['_'.join((eye, units, 'STD', 'y'))] = np.std(rolling_data_y, axis=1)
    measures['_'.join((eye, units, 'STD'))] = np.hypot(measures['_'.join((eye, units, 'STD', 'x'))],
                                                       measures['_'.join((eye, units, 'STD', 'y'))]   
                                                 )
    #STD of PE                                             
    measures['_'.join((eye, units, 'STD_PE', 'x'))] = np.std(rolling_err_x, axis=1)
    measures['_'.join((eye, units, 'STD_PE', 'y'))] = np.std(rolling_err_y, axis=1)
    measures['_'.join((eye, units, 'STD_PE'))] = np.hypot(measures['_'.join((eye, units, 'STD_PE', 'x'))],
                                                          measures['_'.join((eye, units, 'STD_PE', 'y'))]   
                                                 )
    ###
                                              
    ###ACC
    measures['_'.join((eye, units, 'ACC', 'x'))] = np.median(rolling_err_x, axis=1)
    measures['_'.join((eye, units, 'ACC', 'y'))] = np.median(rolling_err_y, axis=1)
    measures['_'.join((eye, units, 'ACC'))] = np.hypot(measures['_'.join((eye, units, 'ACC', 'x'))],
                                                       measures['_'.join((eye, units, 'ACC', 'y'))]   
                                              )
    #Absolute accuracy
    measures['_'.join((eye, units, 'ACC_abs', 'x'))] = np.median(np.abs(rolling_err_x), axis=1)
    measures['_'.join((eye, units, 'ACC_abs', 'y'))] = np.median(np.abs(rolling_err_y), axis=1)
    measures['_'.join((eye, units, 'ACC_abs'))] = np.hypot(measures['_'.join((eye, units, 'ACC_abs', 'x'))],
                                                           measures['_'.join((eye, units, 'ACC_abs', 'y'))]   
                                                  )
    #Fix
    measures['_'.join((eye, units, 'fix', 'x'))] = np.median(rolling_data_x, axis=1)
    measures['_'.join((eye, units, 'fix', 'y'))] = np.median(rolling_data_y, axis=1)
    ###
    
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

    #Data
    rolling_data_x = ma.array(rolling_window(data['_'.join((eye, units, 'x'))], win_size_sample*2),mask=mask_data) 
    rolling_data_y = ma.array(rolling_window(data['_'.join((eye, units, 'y'))], win_size_sample*2),mask=mask_data) 

    #Position error
    err_x = data['_'.join((eye, units, 'x'))] - data[stim_pos_mappings[units]+'x']
    err_y = data['_'.join((eye, units, 'y'))] - data[stim_pos_mappings[units]+'y']
    
    rolling_err_x = ma.array(rolling_window(err_x, win_size_sample*2),mask=mask_data) 
    rolling_err_y = ma.array(rolling_window(err_y, win_size_sample*2),mask=mask_data)
    
    #Time
    rolling_time = ma.array(rolling_window(data['time'], win_size_sample*2),mask=mask_data)
    
    measures['_'.join((eye, units, 'sample_count'))] = np.sum(mask_data, axis=1)
    notmasked_edges=ma.notmasked_edges(rolling_time, axis=1)
    start_times = ma.getdata(rolling_time[notmasked_edges[0][0],notmasked_edges[0][1]])
    end_times = ma.getdata(rolling_time[notmasked_edges[1][0],notmasked_edges[1][1]])
    measures['_'.join((eye, units, 'actual_win_size'))] = end_times-start_times
    
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
    
    #RMS of PE
    isd = np.diff([err_x, err_y], axis=1).T
    
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
    
    measures['_'.join((eye, units, 'RMS_PE', 'x'))] = RMS[0]
    measures['_'.join((eye, units, 'RMS_PE', 'y'))] = RMS[1]    
    measures['_'.join((eye, units, 'RMS_PE'))] = np.hypot(RMS[0], RMS[1])
    ###
    
    
    ###STD 
    STD=[]
    for std in [np.std(rolling_data_x, axis=1), np.std(rolling_data_y, axis=1)]:
        std_tmp = ma.getdata(std)
        mask = ma.getmask(std)
        std_tmp[mask]=np.nan
        STD.append(std_tmp)
    
    measures['_'.join((eye, units, 'STD', 'x'))] = STD[0]
    measures['_'.join((eye, units, 'STD', 'y'))] = STD[1]
    measures['_'.join((eye, units, 'STD'))] = np.hypot(STD[0], STD[1])
    
    #STD of PE 
    STD=[]
    for std in [np.std(rolling_err_x, axis=1), np.std(rolling_err_y, axis=1)]:
        std_tmp = ma.getdata(std)
        mask = ma.getmask(std)
        std_tmp[mask]=np.nan
        STD.append(std_tmp)
    
    measures['_'.join((eye, units, 'STD_PE', 'x'))] = STD[0]
    measures['_'.join((eye, units, 'STD_PE', 'y'))] = STD[1]
    measures['_'.join((eye, units, 'STD_PE'))] = np.hypot(STD[0], STD[1])
                                          
    ###ACC
    ACC=[]
    for acc in [np.median(rolling_err_x, axis=1), np.median(rolling_err_y, axis=1)]:
        acc_tmp = ma.getdata(acc)
        mask = ma.getmask(acc)
        acc_tmp[mask]=np.nan
        ACC.append(acc_tmp)
    
    measures['_'.join((eye, units, 'ACC', 'x'))] = ACC[0]
    measures['_'.join((eye, units, 'ACC', 'y'))] = ACC[1]
    measures['_'.join((eye, units, 'ACC'))] = np.hypot(ACC[0], ACC[1])
    
    #Absolute accuracy
    ACC=[]
    for acc in [np.median(np.abs(rolling_err_x), axis=1), np.median(np.abs(rolling_err_y), axis=1)]:
        acc_tmp = ma.getdata(acc)
        mask = ma.getmask(acc)
        acc_tmp[mask]=np.nan
        ACC.append(acc_tmp)
    
    measures['_'.join((eye, units, 'ACC_abs', 'x'))] = ACC[0]
    measures['_'.join((eye, units, 'ACC_abs', 'y'))] = ACC[1]
    measures['_'.join((eye, units, 'ACC_abs'))] = np.hypot(ACC[0], ACC[1])

    #Fix
    FIX=[]
    for fix in [np.median(rolling_data_x, axis=1), np.median(rolling_data_y, axis=1)]:
        fix_tmp = ma.getdata(fix)
        mask = ma.getmask(fix)
        fix_tmp[mask]=np.nan
        FIX.append(fix_tmp)
    
    measures['_'.join((eye, units, 'fix', 'x'))] = FIX[0]
    measures['_'.join((eye, units, 'fix', 'y'))] = FIX[1]
    ###
    
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
    loss_count = dict()
    data = np.copy(data_wide) 
    for eye in ['left', 'right']:
        #TODO: Filter off-screen, off-pshysical limit samples
        #      eyetribe dynamic et_nan_value 
        trackloss = (data['_'.join((eye, 'gaze_x'))] == et_nan_values[et_model]['x']) | \
                    (data['_'.join((eye, 'gaze_y'))] == et_nan_values[et_model]['y']) | \
                    (np.isnan(data['_'.join((eye, 'gaze_x'))])) | \
                    (np.isnan(data['_'.join((eye, 'gaze_y'))]))
        if et_model == 'dpi':
            trackloss = np.bitwise_or(trackloss, data['status'] < 4.0)
            
        data['_'.join((eye, 'gaze_x'))][trackloss] = fill
        data['_'.join((eye, 'gaze_y'))][trackloss] = fill
        data['_'.join((eye, 'angle_x'))][trackloss] = fill
        data['_'.join((eye, 'angle_y'))][trackloss] = fill
        
        loss_count[eye] = np.sum(trackloss)
    
    if data['eyetracker_mode'][0] == 'Binocular':
        loss_count['avg'] = np.mean((loss_count['right'], loss_count['left']))
    else:
        eye = parseTrackerMode(data['eyetracker_mode'][0])[0]
        loss_count['avg'] = loss_count[eye]
        
    return data, loss_count

def filter_offscreen(data_wide, limit=0, fill=np.nan):
    """
    Off-screen data filter. Replaces invalid samples with /fill/
    
    @author: Raimondas Zemblys
    @email: raimondas.zemblys@humlab.lu.se
    """   
    data = np.copy(data_wide)
    for _dir in ['x', 'y']:
        interval = np.min(data['_'.join(('target_angle', _dir))])-limit, \
                   np.max(data['_'.join(('target_angle', _dir))])+limit
        for eye in parseTrackerMode(data['eyetracker_mode'][0]):
            mask = ma.getmask(ma.masked_outside(data['_'.join((eye, 'angle', _dir))], 
                                               interval[0], interval[1]))
            data['_'.join((eye, 'angle', _dir))][mask]=fill
            data['_'.join((eye, 'gaze', _dir))][mask]=fill
    return data

def plot_data(data, measure='angle', title=None, ylim=None, fname=None, keep=False, stim=None):
    """
    Plots gaze data
    
    @author: Raimondas Zemblys
    @email: raimondas.zemblys@humlab.lu.se
    """
    fig = plt.figure()
    plt.suptitle(title)
    dir_label = dict()
    dir_label['x'] = 'Horizontal position, deg'
    dir_label['y'] = 'Vertical position, deg'
    
    for subplot_ind, _dir in enumerate(['x', 'y']):
        for eye in parseTrackerMode(data['eyetracker_mode'][0]):
            plt.subplot(2,1,subplot_ind+1)
            plt.plot(data['time'], data['_'.join((eye, measure, _dir))])
            plt.plot(data['time'], data['_'.join(('target', measure, _dir))], 'k-')
            plt.xlim(data['time'][0], data['time'][-1])
            plt.ylabel(dir_label[_dir])
            if ylim:
                plt.ylim(ylim[0], ylim[1])
            
            if type(stim) is np.ndarray:
                for stim_row in stim:
                    s = stim_row['_'.join((eye, measure, 'ind'))]
                    e = s+stim_row['_'.join((eye, measure, 'sample_count'))]
                    r = np.arange(s, e, dtype=np.int32)
                    plt.plot(data['time'][r], data['_'.join((eye, measure, _dir))][r], 'r-', linewidth=3)
    
    plt.xlabel('Time, s')
    if fname:   
        plt.savefig(fname, dpi=fig.dpi*4)
    
    if not(keep):
        fig.clf()
        plt.close(fig)
        fig = None 