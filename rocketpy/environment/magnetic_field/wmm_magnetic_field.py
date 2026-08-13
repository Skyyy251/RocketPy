import numpy as np
from pygeomag import GeoMag
from datetime import datetime
from rocketpy.environment.magnetic_field.abstract_magnetic_field import (
    MagneticFieldModel,
)


class WMMMagneticField(MagneticFieldModel):
    def __init__(self):
        self.gm = GeoMag()

    def _to_decimal_year(self, dt):
        year = dt.year
        tz_info = dt.tzinfo
        start_of_year = datetime(year, 1, 1, tzinfo=tz)
        next_year = datetime(year + 1, 1, 1, tzinfo=tz)
        year_fraction = (dt - start_of_year).total_seconds() / (
            next_year - start_of_year
        ).total_seconds()
        return year + year_fraction

    def get_field(self, latitude, longitude, altitude, date):
        alt_km = altitude / 1000.0
        decimal_year = self._to_decimal_year(date)

        res = self.gm.calculate(latitude, longitude, alt_km, decimal_year)

        field_vector = np.array([rex.x, res.y, res.z]) * 1e-9

        return field_vector
