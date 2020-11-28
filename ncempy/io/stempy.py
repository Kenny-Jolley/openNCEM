"""
This module provide an interface for loading HDF5 files output from stempy for the 4D Camera.py

The files are sparse datasets of counted 4D Camera frames and usually used for 4D STEM


"""

from pathlib import Path
from collections import namedtuple

import h5py
import numpy as np

class fileSTEMPY:
    """ Class to represent stempy counted datasets

    Attributes
    ----------

    Methods
    -------



    """

    def __init__(self, filename):
        """
        Parameters
        ----------
        filename : str or pathlib.Path

        """

        # necessary declarations in case something goes bad
        self.file_hdl = None

        # convenience handles to access the data in the emd file, everything can as well be accessed using the file_hdl
        self.scan_dimensions = None
        self.scan_positions = None
        self.filename = filename
        self.frame_dimensions = None
        self.data = None

        # check filename type, change to string
        if isinstance(filename, str):
            pass
        elif isinstance(filename, Path):
            filename = str(filename)
        else:
            raise TypeError('Filename is supposed to be a string or pathlib.Path')

        try:
            self.file_hdl = h5py.File(filename, 'r')
        except IOError:
            print('Error opening file: "{}"'.format(filename))
            raise

        if self.file_hdl:
            self.data = self.file_hdl['electron_events/frames']
            self.frame_dimensions = (self.data.attrs['Ny'], self.data.attrs['Nx'])
            scan_grp = self.file_hdl['electron_events/scan_position']
            self.scan_dimensions = (scan_grp.attrs['Ny'], scan_grp.attrs['Nx']) # row col
            self.scan_positions = scan_grp

    def __del__(self):
        """Destructor for HDF5 file object.

        """
        # close the file
        # if(not self.file_hdl.closed):
        self.file_hdl.close()

    def __enter__(self):
        """Implement python's with statement

        """
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Implement python's with statement
        and close the file via __del__()
        """
        self.__del__()
        return None

    def getDataset(self, reshape=False):
        """Load the entire dataset into memory as a ragged numpy array. The array can be reshaped to three dimensionas
        with axes [scanX, scanY, event_location]

        Parameters
        ----------
        reshape : bool, default False
            If True then the data is reshaped to match the scan dimensions. If False then the first axis is raveled.

        """
        out = self.data[:]

        if reshape:
            out.reshape(self.scan_dimensions)

        return out

    def getStempy(self):
        """ Return the data set as a named tuple like which is compatible with stempy.

        Returns
        -------
        : namedtuple
            The data, scan dimension and frame dimension as a namedtyple compatible with stempy
        """

        ret = namedtuple('ElectronCountedData',
                         ['data', 'scan_dimensions', 'frame_dimensions'])
        ret.data = self.data[:]
        ret.scan_dimensions = self.scan_dimensions
        ret.frame_dimensions = self.frame_dimensions

        return ret

    def getNamedtuple(self):
        """Same as getStempy."""
        return self.getStempy()

    def getArray(self):
        """Return a separate class which allows simple slicing and other normal interactions with the ragged array.
        Not implemented Yet

        """
        pass

    def getMemmap(self):
        """Return the data but keep it on disk.
        Returns
        -------
        : h5py.dataset
            The h5py dataset on disk.
        """

    def sum_sparse(self, start=0, end=-1):
        """Return a dense frame summed from all of the events in the range specified by start and end.

        Parameters
        ----------
        start : int
            The first frame to use
        end : int
            The last frame to use

        """
        dp = np.zeros((self.frame_dimensions[0]*self.frame_dimensions[1], ), '<u4')
        for ii, ev in enumerate(self.data[start:end]):
            dp[ev] += 1
        dp = dp.reshape(self.frame_dimensions)
        return dp

    def com_sparse(self):
        """Calculate the center of mass of every frame using a sparse algorithm.
        """
        com2 = np.zeros((2, self.scan_dimensions[0]*self.scan_dimensions[1]), np.float32)
        for ii, ev in enumerate(self.data):
            if len(ev) > 0:
                x, y = np.unravel_index(ev, (576, 576))
                mm = len(ev)
                comx = np.sum(x)/mm
                comy = np.sum(y)/mm
                com2[:, ii] = (comy, comx)
            else:
                com2[:, ii] = (np.nan, np.nan)

        com2 = com2.reshape((2, self.scan_dimensions[0], self.scan_dimensions[1]))
        return com2


def stempyReader(fname):
    """Read in the stempy hdf5 data and returns as a dictionary

    Returns
    -------
    : dict
        A dictionary of the data and metadata with keys data, scan_dimensions, and frame dimensions.
    """
    out = {}
    with fileSTEMPY(fname) as f0:
        out['data'] = f0.data[:]
        out['scan_dimensions'] = f0.scan_dimensions
        out['frame_dimensions'] = f0.frame_dimensions
    return out
