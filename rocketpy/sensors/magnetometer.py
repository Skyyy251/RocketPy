import numpy as np
import matplotlib.pyplot as plt

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

        def _mat_to_np(M):
            return np.array(
                [[M.xx, M.xy, M.xz], [M.yx, M.yy, M.yz], [M.zx, M.zy, M.zz]],
                dtype=float,
            )

        # Vektoren: 3x1 -> numpy
        def _vec_to_np(v):
            return np.array([v[0], v[1], v[2]], dtype=float)

        # Debug
        R_inertial_to_sensor = _mat_to_np(inertial_to_sensor)
        R_sensor_to_inertial = R_inertial_to_sensor.T  # numpy-Transpose, safe

        self._debug_orientations.append((time, R_sensor_to_inertial))
        self._debug_B_inertial.append((time, _vec_to_np(B_inertial)))
        self._debug_B_sensor.append((time, _vec_to_np(B_sensor)))

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

    def plot_debug_axes(self):
        """Plot sensor axis orientation trajectories in inertial frame and components over time."""
        times = np.array([t for t, _ in self._debug_orientations])
        R = np.array([m for _, m in self._debug_orientations])  # shape (N, 3, 3)

        # Spalten = Sensor-Achsen im inertialen Frame
        x_axis = R[:, :, 0]
        y_axis = R[:, :, 1]
        z_axis = R[:, :, 2]

        # --- 3D-Trajektorie der Achsen-Spitzen ---
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(x_axis[:, 0], x_axis[:, 1], x_axis[:, 2], label="Sensor x", color="r")
        ax.plot(y_axis[:, 0], y_axis[:, 1], y_axis[:, 2], label="Sensor y", color="g")
        ax.plot(z_axis[:, 0], z_axis[:, 1], z_axis[:, 2], label="Sensor z", color="b")

        # Start- und Endpunkte markieren
        for ax_arr, color in [(x_axis, "r"), (y_axis, "g"), (z_axis, "b")]:
            ax.scatter(*ax_arr[0], color=color, marker="o", s=60)
            ax.scatter(*ax_arr[-1], color=color, marker="*", s=120)

        ax.set_xlabel("X_inertial")
        ax.set_ylabel("Y_inertial")
        ax.set_zlabel("Z_inertial")
        ax.set_title(f"{self.name} Sensorachsen im inertialen Frame")
        ax.legend()
        ax.set_box_aspect([1, 1, 1])
        plt.tight_layout()
        plt.show()

        # --- 2D: Achsenkomponenten über Zeit ---
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        for ax_i, axis, name in zip(axes, [x_axis, y_axis, z_axis], ["x", "y", "z"]):
            ax_i.plot(times, axis[:, 0], label=f"{name}·X_inertial")
            ax_i.plot(times, axis[:, 1], label=f"{name}·Y_inertial")
            ax_i.plot(times, axis[:, 2], label=f"{name}·Z_inertial")
            ax_i.set_ylabel(f"Sensor-{name}-Achse")
            ax_i.grid(True, ls="--", alpha=0.5)
            ax_i.legend(loc="best", ncol=3, fontsize=9)
        axes[-1].set_xlabel("Time [s]")
        fig.suptitle(f"{self.name} Sensorachsen-Komponenten")
        plt.tight_layout()
        plt.show()
