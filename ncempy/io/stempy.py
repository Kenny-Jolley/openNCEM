"""
This module provide an interface for loading HDF5 files output from stempy for the 4D Camera.py

The files are sparse datasets of counted 4D Camera frames and usually used for 4D STEM


"""

from pathlib import Path

import h5py
import numpy

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
            self.scan_dimensions = (scan_grp.attrs['Ny'], scan_grp.attrs['Nx'])
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

    def getDataset(self):
        """Load the entre dataset into memory."""

    def get_as_namedtuple(self):
        """Stempy typically operates on data as a namedtuple. Recreate such a dataset like in stempy"""
        pass

    def get_as_sparse_array(self):
        """Return a separate class which allows simple slicing and other normal interactions with the ragged array"""
        pass