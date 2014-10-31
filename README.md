#EDQ Result Files Conversion Script
This folder contains the script used to convert eye samples from ioHub DataStore HDF5 files into a "wide" format, where each eye sample row contains information about the participant, session, eye tracker and display equipment used, in addition to the eye sample data itself. 

Part of the conversion process includes only selecting samples that have a time stamp within the period a fixation target was visible on the screen.

##Results Files Converted
The following eye tracker model data is handled by this script:

    dpi*
    eyefollower 
    eyelink 
    eyetribe 
    hispeed1250 
    hispeed240
    red250 
    red500 
    redm 
    t60xl 
    tx300 
    x2

  \* For calibration details, see _**DPI calibration**_ below

##Results Files NOT Converted

The following eye tracker model data has not been converted by this script because the eye data was not saved by ioHub:

    asl
    positivescience
    smietg
    smihed
    tobiiglasses

##Converted Data File Types
The data files are converted into two file types:

    *.txt - tab delimited plain text files, open with anything, including Excel.
    *.npy - binary NumPy files. Open using numpy.load() http://docs.scipy.org/doc/numpy/reference/generated/numpy.load.html.
      
##Converted File Format
Each saved file represents the data collected from 1 - N sessions / runs of the test, with one participant using one of the tested eye tracker models. Most files only have one session, while a minority have > 1 session saved to the same file. Data from different sessions within a single output file can be grouped by the session_id column.

Each converted file contain N rows, where N is the number of eye samples that occurred within the time period the fixation target was visible at one of the 51 target positions presented.

Each converted file has the following columns. In the case of the .txt type, a header row is written as the first line in the file, providing the name of each column. For the .npy type files, the column names provided here match the  element names of each structured numpy array.

	* subject_id : ID assigned to a specific participant. 
	* display_refresh_rate : Hz update rate of display.
	* eyetracker_model : The unique label given to each eye tracker model tested.
	* dot_deg_sz : Approximate size of the target in visual degrees.
	* eyetracker_sampling_rate : Sampling rate of the eye tracker being used, in Hz.
	* eyetracker_mode : The eyes that were being tracked (Binocular, Left Eye, Right Eye)
	* fix_stim_center_size_pix : The pixel size of the fixation target inner dot
	* operator : Alpha character code assigned to a specific operator performing data collection.
	* eyetracker_id : A unique integer ID assigned to each eye tracker.
	* display_width_pix : The horizontal resolution of the display in pixels.
	* display_height_pix : The vertical resolution of the display in pixels.
	* screen_width : The horizontal length of the display in mm.
	* screen_height : The vertical length of the display in mm.
	* eye_distance : The approximate eye to display center distance in mm.
	* SESSION_ID : The ioHub session ID assigned within a given hdf5 file.
	* trial_id : target display sequence 
	* TRIAL_START : The time (in seconds) the fixation target started to be presented at a the given target location.
	* TRIAL_END : The time (in seconds) the fixation target was removed from the given target location.
	* posx : The horizontal position of the target in pixels. NOTE: The pixel origin ( 0, 0) is the center of the screen.
	* posy : The vertical position of the target in pixels. NOTE: The pixel origin ( 0, 0) is the center of the screen.
	* dt : The duration the target was presented for (actual presentation time will be a multiple of the display_refresh_rate).
	* ROW_INDEX : The ID of the target position being displayed.
	* BLOCK : experiment block: FS – Fixate-Saccade, SP – smooth pursuit, IMG – picture viewing task, TXT – sentence reading task, DPICAL – dpi calibration targets
	* session_id : Same as SESSION_ID.
	* device_time : The time of the sample as reported by the eye tracking device.
	* time : The time of the sample after being converted to the ioHub time base (in seconds).
	* left_gaze_x : Left eye horizontal position in pixels. * 0.0 is screen center.
	* left_gaze_y : Left eye vertical position in pixels. * 0.0 is screen center.
	* left_pupil_measure1 : Left eye pupil size / diameter. * units are eye tracker model specific.
	* right_gaze_x : Right eye horizontal position in pixels. * 0.0 is screen center.
	* right_gaze_y : Right eye vertical position in pixels. * 0.0 is screen center.
	* right_pupil_measure1 : Right eye pupil size / diameter. * units are eye tracker model specific.
	* status : ioHub status field for the sample. 0 = no issues; > 0 = left and/or right eye data is missing.
	* target_angle_x :  Horizontal position of target in visual degrees. 0 is screen center.
	* target_angle_y :  Vertical position of target in visual degrees. 0 is screen center.
	* left_angle_x :  Left eye horizontal position in visual degrees. 0 is screen center.
	* left_angle_y :  Left eye vertical position in visual degrees. 0 is screen center.
	* right_angle_x : Right eye horizontal position in visual degrees. 0 is screen center.
	* right_angle_y : Right eye vertical position in visual degrees. 0 is screen center.

##Missing Position Data Values
The value saved to the eye position gaze x and y columns is dependant on the eye tracker model the data as collected from:

    eyefollower: 0
    eyelink : 0
    eyetribe : -840 for x, 
                525 for y
    hispeed1250 : 0 
    hispeed240 : 0
    red250 : 0
    red500 : 0
    redm : 0
    t60xl : -1
    tx300 : -1
    x2 : -1
    dpi : -1000

##DPI calibration
DPI eye position data (raw analog data) is converted into screen pixel positions using second degree polynomial mapping. Polynomial regression is used to find coeficients ax, ay, bx, by, cx, cy, dx, dy, ex, ey of:

    X = ax v1^2+bx v2^2+cx v1v2+ dx v1+ex v2+fx
    Y = ay v1^2+by v2^2+cy v1v2+ dy v1+ey v2+fy
This mapping needs at least 16 calibration points. The 16 target points selected are uniformly distributed on the screen in order to get a good mapping of the whole screen area. Gaze _fixations_ were identified using running window of 175ms as median gaze position within each minRMS window. 
If it is not available to get _fixation_ for any of predefined calibration points due to trackloss, calibration point is replaced with randomly selected one. 
By default 16 calibration points are used to perform calibration, but there is a possibility to calibrate using less points (see _**Script configuration**_ section below)

#Running Conversion Script Locally


To run the script you will need the software listed in the Installation section, as well as the EDQ HDF5 files, organized as described in the Expected Folder / File Structure section. The archive file containing the data is already in this file structure, so you should not have to rearrange data files.  

##Starting the Conversion Program

    Open the run_conversion.py file in your python IDE of choice and run it

or

Open a command prompt / terminal window, and type the following:

    cd [path to the directory this file is in]
    python.exe run_conversion.py

##Expected Folder / File Structure 

The HDF5 files must be in a folder called **_hdf5_data_root_** (See *Script configuration* below). The **_hdf5_data_root_** folder must have the following structure:

    hdf5_data_root
       |
       |--[eye tracker name 1]
       |     |
       |     |--[date1]
       |     |    |
       |     |    |-- events_XX.hdf5
       |     |    |-- ....
       |     |    |-- events_NN.hdf5
       |     |
       |     |-- ....
       |     |
       |     |--[dateN]
       |          |
       |          |-- events_XX.hdf5
       |          |-- ....
       |          |-- events_NN.hdf5
       |
       |-- ......
       |
       |   
       |--[eye tracker name N]   
       |     |
       |     |-- .....
       |     |    |
       |     |    |-- ......

##Script configuration

Script has number of adjustable settings. Open the run_conversion.py file in your python IDE of choice and modify following variables according to your needs:

    * INCLUDE_TRACKERS : remove tracker names, that you do not want to convert
    
    * INCLUDE_SUB : 'ALL' will convert recordings from all subjects, or specify subjects IDs you want to convert (in this case, variable should be a list). 
    
    * INPUT_FILE_ROOT: path to root folder of recordings
    
    * OUTPUT_FOLDER : output folder root. Tail of the output folder path will be set to: rev_[current-version-hash][local-source-same], where
        [current-version-hash]: the short form hash of the github project source version at the time the script is run
        [local-source-same]: State of local project code source being run compared to project github master.
            - Empty string if local matches projects remote master HEAD
            - '_unsynced' string if local project has changes, or has not been synced, with github remote master.
            
    * SAVE_NPY : set to True if you want to save data as binary NumPy files. See Converted Data File Types. Set to False if not required
    
    * SAVE_TXT : set to True if you want to save data as tab delimited plain text files. See Converted Data File Types. Set to False if not required
    
    * BLOCKS_TO_EXPORT : list of blocks to be exported: FS – Fixate-Saccade, SP – smooth pursuit, IMG – picture viewing task, TXT – sentence reading task, DPICAL – dpi calibration targets
    
    * FIX_DPI_CAL : set to True only if you're dealing with 2014 Apr-May EDQ recordings from Lund University, Humanities Laboratory. This fixes a bug with incorrect ROW_INDEX values in DPICAL block
    
    * calibrate_dpi : set to True if DPI data should be calibrated. If set to False, raw analog values are exported
    
    * min_calibration_points : set the lower boundary of calibration points to be used when calibrating DPI. If it is not possible to calculate positions of 16 _fixations_, less calibration points can be used.