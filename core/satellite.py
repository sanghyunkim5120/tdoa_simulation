import numpy as np


EARTH_RADIUS = 6371.0  # km
SPEED_OF_LIGHT = 299792.458  # km/s


class Satellite:
    def __init__(self, sat_id: int, orbit_radius: float = None,
                 inclination: float = None, raan: float = None,
                 initial_angle: float = None, angular_speed: float = None):
        self.sat_id = sat_id
        self.orbit_radius = orbit_radius or (EARTH_RADIUS + np.random.uniform(1000, 3000))
        self.inclination = inclination if inclination is not None else np.random.uniform(0, 60)
        self.raan = raan if raan is not None else np.random.uniform(0, 360)
        self.angle = initial_angle if initial_angle is not None else np.random.uniform(0, 360)
        self.angular_speed = angular_speed if angular_speed is not None else \
            np.random.uniform(0.3, 0.8) * np.random.choice([-1, 1])
        self.color = _random_color()

    def update(self, dt: float):
        self.angle = (self.angle + self.angular_speed * dt) % 360

    def get_position_3d(self) -> np.ndarray:
        inc = np.radians(self.inclination)
        raan = np.radians(self.raan)
        angle = np.radians(self.angle)
        r = self.orbit_radius

        x_orb = r * np.cos(angle)
        y_orb = r * np.sin(angle)

        # 궤도면 → ECEF
        x = (np.cos(raan) * x_orb - np.sin(raan) * np.cos(inc) * y_orb)
        y = (np.sin(raan) * x_orb + np.cos(raan) * np.cos(inc) * y_orb)
        z = np.sin(inc) * y_orb
        return np.array([x, y, z])

    def get_position_2d(self) -> np.ndarray:
        pos3d = self.get_position_3d()
        return np.array([pos3d[0], pos3d[1]])

    def get_orbit_path_3d(self, steps: int = 100) -> tuple:
        inc = np.radians(self.inclination)
        raan = np.radians(self.raan)
        angles = np.linspace(0, 2 * np.pi, steps)
        r = self.orbit_radius

        xs, ys, zs = [], [], []
        for angle in angles:
            x_orb = r * np.cos(angle)
            y_orb = r * np.sin(angle)
            x = np.cos(raan) * x_orb - np.sin(raan) * np.cos(inc) * y_orb
            y = np.sin(raan) * x_orb + np.cos(raan) * np.cos(inc) * y_orb
            z = np.sin(inc) * y_orb
            xs.append(x)
            ys.append(y)
            zs.append(z)
        return xs, ys, zs


class UserPosition:
    def __init__(self, lat: float = None, lon: float = None):
        self.lat = lat if lat is not None else np.random.uniform(-60, 60)
        self.lon = lon if lon is not None else np.random.uniform(-180, 180)

    def get_position_3d(self) -> np.ndarray:
        lat = np.radians(self.lat)
        lon = np.radians(self.lon)
        r = EARTH_RADIUS
        x = r * np.cos(lat) * np.cos(lon)
        y = r * np.cos(lat) * np.sin(lon)
        z = r * np.sin(lat)
        return np.array([x, y, z])

    def get_position_2d(self) -> np.ndarray:
        pos3d = self.get_position_3d()
        return np.array([pos3d[0], pos3d[1]])


def _random_color():
    colors = [
        '#00FF88', '#FF6B35', '#4ECDC4', '#FFE66D',
        '#A8DADC', '#FF8FA3', '#C77DFF', '#80FFDB'
    ]
    return np.random.choice(colors)
