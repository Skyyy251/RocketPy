from abc import ABC, abstractmethod
import numpy as numpy


class MagneticFieldModel(ABC):
    """Abstract model for Earth's magnetic field"""

    units = "T"

    @abstractmethod
    def get_field(self, latitude, longitude, altitude, date):
        """Calculate the magnetic field.

        Parameters
        ----------
        latitude : float
            Latitude in degrees.

        longitude : float
            Longitude in degrees.

        altitude : float
            Altitude above sea level in meters.

        date : datetime
            Date and time for which the magnetic field is calculated.

        Returns
        -------
        numpy.ndarray
            Magnetic field vector [Bx, By, Bz] in Tesla.
        """
        raise NotImplementedError
