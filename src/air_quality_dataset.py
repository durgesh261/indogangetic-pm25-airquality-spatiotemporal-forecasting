"""
Indo-Gangetic Plain Air Quality & Satellite Fire Dataset Generator.
Synthesizes multi-station CPCB ground monitoring data, NASA satellite active fire counts,
and meteorological wind fields across the Delhi-NCR, Punjab, Haryana, and UP corridor.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any


# 15 Representative ground monitoring stations across the Indo-Gangetic Basin
STATIONS = [
    {"id": 0, "name": "Delhi - Anand Vihar", "state": "Delhi", "lat": 28.65, "lon": 77.31, "is_delhi_ncr": True},
    {"id": 1, "name": "Delhi - Punjabi Bagh", "state": "Delhi", "lat": 28.67, "lon": 77.13, "is_delhi_ncr": True},
    {"id": 2, "name": "Delhi - ITO", "state": "Delhi", "lat": 28.63, "lon": 77.24, "is_delhi_ncr": True},
    {"id": 3, "name": "Delhi - RK Puram", "state": "Delhi", "lat": 28.56, "lon": 77.17, "is_delhi_ncr": True},
    {"id": 4, "name": "Noida - Sector 62", "state": "Uttar Pradesh", "lat": 28.62, "lon": 77.36, "is_delhi_ncr": True},
    {"id": 5, "name": "Ghaziabad - Vasundhara", "state": "Uttar Pradesh", "lat": 28.66, "lon": 77.38, "is_delhi_ncr": True},
    {"id": 6, "name": "Gurugram - Vikas Sadan", "state": "Haryana", "lat": 28.45, "lon": 77.03, "is_delhi_ncr": True},
    {"id": 7, "name": "Faridabad - Sector 16A", "state": "Haryana", "lat": 28.41, "lon": 77.32, "is_delhi_ncr": True},
    {"id": 8, "name": "Amritsar - Golden Temple", "state": "Punjab", "lat": 31.62, "lon": 74.87, "is_delhi_ncr": False},
    {"id": 9, "name": "Ludhiana - Punjab Agri Univ", "state": "Punjab", "lat": 30.90, "lon": 75.81, "is_delhi_ncr": False},
    {"id": 10, "name": "Patiala - Model Town", "state": "Punjab", "lat": 30.34, "lon": 76.38, "is_delhi_ncr": False},
    {"id": 11, "name": "Ambala - Poly Tech", "state": "Haryana", "lat": 30.38, "lon": 76.78, "is_delhi_ncr": False},
    {"id": 12, "name": "Karnal - Sector 12", "state": "Haryana", "lat": 29.69, "lon": 76.98, "is_delhi_ncr": False},
    {"id": 13, "name": "Panipat - Sanjay Colony", "state": "Haryana", "lat": 29.39, "lon": 76.97, "is_delhi_ncr": False},
    {"id": 14, "name": "Lucknow - Gomti Nagar", "state": "Uttar Pradesh", "lat": 26.85, "lon": 80.99, "is_delhi_ncr": False}
]


class IndoGangeticAirQualityDataset:
    """
    Synthesizes multi-station hourly observations of PM2.5, meteorological fields,
    and upstream agricultural fire counts over a 120-day post-monsoon winter period (Oct-Jan).
    """

    def __init__(
        self,
        num_days: int = 60,
        seq_len: int = 48,  # 2 days lookback (48 hours)
        horizons: Tuple[int, int, int] = (24, 48, 72),  # Forecast horizons in hours
        random_state: int = 42
    ):
        self.num_days = num_days
        self.total_hours = num_days * 24
        self.seq_len = seq_len
        self.horizons = horizons
        self.num_stations = len(STATIONS)
        self.rng = np.random.default_rng(random_state)

    def generate(self) -> Dict[str, Any]:
        """
        Generates full spatiotemporal dataset.
        Returns:
            X_seq: [N_samples, num_stations, seq_len, num_features]
            Y_targets: [N_samples, num_stations, 3] (24h, 48h, 72h future PM2.5)
            fire_intensity: [total_hours] Upstream satellite fire counts
            dist_matrix: [num_stations, num_stations] Inter-station distance matrix
            station_info: List of station metadata
        """
        t = np.arange(self.total_hours)

        # 1. Stubble burning seasonal fire cycle (peaks late October - mid November, days 25 to 60)
        peak_day = 42.0  # ~Nov 12
        day_indices = t / 24.0
        fire_base = 3500.0 * np.exp(-((day_indices - peak_day) ** 2) / (2 * (10.0 ** 2)))
        fire_noise = self.rng.gamma(shape=2.0, scale=80.0, size=self.total_hours)
        satellite_fire_counts = np.maximum(50.0, fire_base + fire_noise)

        # 2. Meteorological fields (Northwesterly wind brings smoke down to Delhi)
        # Wind direction: 300-330 degrees (North-West)
        wind_dir_rad = np.radians(self.rng.normal(loc=315.0, scale=25.0, size=self.total_hours))
        wind_speed = self.rng.uniform(1.5, 4.5, size=self.total_hours)  # Calm winter winds (m/s)
        wind_u = wind_speed * np.sin(wind_dir_rad)
        wind_v = wind_speed * np.cos(wind_dir_rad)

        # Temperature & Boundary Layer Height (inversion drops at night, trapping smoke)
        diurnal_cycle = np.sin(2 * np.pi * t / 24.0 - np.pi / 2)  # Low at night, peak at 2 PM
        temp_ambient = 18.0 + 8.0 * diurnal_cycle - 0.05 * day_indices  # Cooling into winter
        boundary_layer_height = 400.0 + 600.0 * np.maximum(0.0, diurnal_cycle)  # 400m night to 1000m day
        relative_humidity = 55.0 - 25.0 * diurnal_cycle + self.rng.normal(0, 5, size=self.total_hours)
        relative_humidity = np.clip(relative_humidity, 20.0, 95.0)

        # 3. Simulate PM2.5 per station based on physics:
        # PM2.5 = Urban Baseline + Upstream Fire Plume (if wind is NW) + Inversion Trapping Factor
        station_pm25 = np.zeros((self.num_stations, self.total_hours), dtype=np.float32)

        for s_idx, station in enumerate(STATIONS):
            # Base urban emissions (higher in dense Delhi-NCR)
            urban_base = 80.0 if station["is_delhi_ncr"] else 55.0
            urban_noise = self.rng.normal(0, 15, size=self.total_hours)

            # Upstream fire impact: Punjab stations feel it first; Delhi feels it 24-48h later
            # Distance from Punjab fire centroid (approx lat 31.0, lon 75.5)
            dist_from_punjab = np.sqrt((station["lat"] - 31.0)**2 + (station["lon"] - 75.5)**2) * 111.0  # km
            transport_delay_hrs = int(dist_from_punjab / 15.0)  # ~15 km/h wind transport
            
            delayed_fires = np.roll(satellite_fire_counts, transport_delay_hrs)
            delayed_fires[:transport_delay_hrs] = 100.0

            # Inversion factor: lower boundary layer traps pollutants
            inversion_multiplier = 800.0 / np.clip(boundary_layer_height, 250.0, 1500.0)

            # Plume contribution (scaled by northwesterly wind component)
            nw_wind_factor = np.clip(-wind_v / (wind_speed + 1e-4), 0.0, 1.0)
            fire_plume = (delayed_fires * 0.045) * nw_wind_factor * inversion_multiplier

            # Total PM2.5 concentration
            pm25 = (urban_base + urban_noise) * inversion_multiplier + fire_plume
            station_pm25[s_idx, :] = np.clip(pm25, 20.0, 850.0)  # CPCB realistic range

        # 4. Construct feature tensors: [total_hours, num_stations, 6 features]
        # Features: [PM2.5, wind_speed, wind_direction, temp, humidity, upstream_fires]
        all_features = np.zeros((self.total_hours, self.num_stations, 6), dtype=np.float32)

        for s_idx in range(self.num_stations):
            all_features[:, s_idx, 0] = station_pm25[s_idx, :]
            all_features[:, s_idx, 1] = wind_speed
            all_features[:, s_idx, 2] = wind_dir_rad
            all_features[:, s_idx, 3] = temp_ambient
            all_features[:, s_idx, 4] = relative_humidity
            all_features[:, s_idx, 5] = satellite_fire_counts

        # Normalize features
        feat_mean = np.mean(all_features, axis=(0, 1), keepdims=True)
        feat_std = np.std(all_features, axis=(0, 1), keepdims=True) + 1e-5
        norm_features = (all_features - feat_mean) / feat_std

        # 5. Extract multi-horizon sequence samples
        # Sample every 6 hours to create rich dataset
        sample_indices = range(self.seq_len, self.total_hours - max(self.horizons), 6)
        x_samples = []
        y_samples = []

        for idx in sample_indices:
            x_window = norm_features[idx - self.seq_len:idx]  # [seq_len, num_stations, 6]
            x_window = np.transpose(x_window, (1, 0, 2))      # [num_stations, seq_len, 6]

            # Future PM2.5 targets at 24h, 48h, 72h
            y_window = np.zeros((self.num_stations, 3), dtype=np.float32)
            for h_idx, h in enumerate(self.horizons):
                y_window[:, h_idx] = station_pm25[:, idx + h]

            x_samples.append(x_window)
            y_samples.append(y_window)

        x_arr = np.array(x_samples, dtype=np.float32)  # [N, num_stations, seq_len, 6]
        y_arr = np.array(y_samples, dtype=np.float32)  # [N, num_stations, 3]

        # Inter-station pairwise distance matrix (in km)
        coords = np.array([[s["lat"], s["lon"]] for s in STATIONS])
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1)) * 111.0  # approximate km

        # Split train / val / test (70% / 15% / 15% temporal split)
        n_samples = len(x_arr)
        n_train = int(0.70 * n_samples)
        n_val = int(0.15 * n_samples)

        return {
            "x_train": torch.from_numpy(x_arr[:n_train]),
            "y_train": torch.from_numpy(y_arr[:n_train]),
            "x_val": torch.from_numpy(x_arr[n_train:n_train + n_val]),
            "y_val": torch.from_numpy(y_arr[n_train:n_train + n_val]),
            "x_test": torch.from_numpy(x_arr[n_train + n_val:]),
            "y_test": torch.from_numpy(y_arr[n_train + n_val:]),
            "dist_matrix": torch.from_numpy(dist_matrix).float(),
            "stations": STATIONS,
            "raw_pm25": station_pm25,
            "fire_counts": satellite_fire_counts,
            "feat_mean": feat_mean.squeeze(),
            "feat_std": feat_std.squeeze(),
            "horizons": self.horizons
        }
