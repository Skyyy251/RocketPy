from rocketpy.environment.magnetic_field.abstract_magnetic_field import (
    MagneticFieldModel,
)
import numpy as np


class ConstantMagneticField(MagneticFieldModel):  #
    """Constant magnetic field model"""

    def __init__(self, field):
        self.field = np.asarray(field, dtype=format)

        if self.field.shape != (3,):
            raise ValueError("Magnetic field must be a 3-element vector.")

    def get_field(self, latitude, longitude, altitude, date):
        return self.field
