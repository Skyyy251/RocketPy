import numpy as np

from ..mathutils.vector_matrix import Matrix, Vector
from ..prints.sensors_prints import _InertialSensorPrints
from ..sensors.sensor import InertialSensor


class Magnetometer(InertialSensor):
    units = "T"

    def __init__(
        self,
        sampling_rate,
        orientation=(0, 0, 0),
        measurement_range=np.inf,
        resolution=0,
        noise_density=0,
        noise_variance=1,
        random_walk_density=0,
        random_walk_variance=1,
        constant_bias=0,
        operating_temperature=298.15,
        temperature_bias=0,
        temperature_scale_factor=0,
        cross_axis_sensitivity=0,
        name="Magnetometer",
        seed=None,
    ):
        super().__init__(
            sampling_rate,
            orientation,
            measurement_range=measurement_range,
            resolution=resolution,
            noise_density=noise_density,
            noise_variance=noise_variance,
            random_walk_density=random_walk_density,
            random_walk_variance=random_walk_variance,
            constant_bias=constant_bias,
            operating_temperature=operating_temperature,
            temperature_bias=temperature_bias,
            temperature_scale_factor=temperature_scale_factor,
            cross_axis_sensitivity=cross_axis_sensitivity,
            name=name,
            seed=seed,
        )

        self.prints = _InertialSensorPrints(self)
        self._debug_calls = []
        self._debug_orientations = []
        self._debug_B_inertial = []
        self._debug_B_sensor = []

    def measure(self, time, **kwargs):
        """Measure the Earth's magnetic field.

        Parameters
        ----------
        time : float
            Current simulation time.

        kwargs : dict
            u : Rocket state vector
            environment : Environment object
        """
        self._debug_calls.append(time)

        u = kwargs["u"]
        environment = kwargs["environment"]

        # Get the eraths magnetic field in Interial frame

        if environment.magnetic_field is None:
            raise ValueError("Environment has no magnetic field model")

        B_inertial = Vector(
            environment.magnetic_field.get_field(
                latitude=environment.latitude,
                longitude=environment.longitude,
                altitude=u[2],
                date=environment.datetime_date,
            )
        )

        # Rotate inertial -> sensor frame

        inertial_to_sensor = (
            self._total_rotation_sensor_to_body
            @ Matrix.transformation(u[6:10]).transpose
        )

        B_sensor = inertial_to_sensor @ B_inertial

        # --- Debug: Orientierung und Felder speichern ---

        R_sensor_to_inertial = inertial_to_sensor.transpose()
        self._debug_orientations.append((time, np.array(R_sensor_to_inertial)))

        self._debug_B_inertial.append((time, np.array(B_inertial)))
        self._debug_B_sensor.append((time, np.array(B_sensor)))

        # Apply noice

        B_sensor = self.apply_noise(B_sensor)
        B_sensor = self.apply_temperature_drift(B_sensor)
        B_sensor = self.quantize(B_sensor)

        # Store measurement_range
        self.measurement = tuple([*B_sensor])
        self._save_data((time, *B_sensor))

    def export_measured_data(self, filename, file_format="csv"):
        """Export measured magnetic field."""

        self._generic_export_measured_data(
            filename=filename,
            file_format=file_format,
            data_labels=("t", "Bx", "By", "Bz"),
        )

    def to_dict(self, **kwargs):
        return super().to_dict(**kwargs)

    @classmethod
    def from_dict(cls, data):
        return cls(
            sampling_rate=data["sampling_rate"],
            orientation=data["orientation"],
            measurement_range=data["measurement_range"],
            resolution=data["resolution"],
            noise_density=data["noise_density"],
            noise_variance=data["noise_variance"],
            random_walk_density=data["random_walk_density"],
            random_walk_variance=data["random_walk_variance"],
            constant_bias=data["constant_bias"],
            operating_temperature=data["operating_temperature"],
            temperature_bias=data["temperature_bias"],
            temperature_scale_factor=data["temperature_scale_factor"],
            cross_axis_sensitivity=data["cross_axis_sensitivity"],
            name=data["name"],
            seed=data.get("seed"),
        )
