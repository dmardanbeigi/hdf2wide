__author__ = 'Sol'
"""
Copied code from iohub
(trying to keep dependencies low for this conversion script)
"""

import numpy as np
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

