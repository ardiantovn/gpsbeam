import numpy as np
from loguru import logger
from numpy.typing import NDArray
from typing import Tuple

class GpsPrep:
    def __init__(self):
        logger.debug("GpsPrep Initialized.")

    def read(self, raw_scenario_folder: str, fpath: str) -> NDArray:
        """Read GPS data from file and 
        convert first two rows(lat,lon) to float."""
        gps_data = np.loadtxt(f'{raw_scenario_folder}/{fpath[1:]}', dtype=str)
        return gps_data.astype(float)

    def minmax_norm(self, latlon_arr: NDArray, min_val: float=None, max_val: float=None) -> NDArray:
        """Perform min-max normalization 
        on latitude and longitude data."""
        latlon = np.array(latlon_arr, dtype=float)
        if min_val is None:
            min_vals = latlon.min(axis=0)
        else:
            min_vals = min_val
        if max_val is None:
            max_vals = latlon.max(axis=0)
        else:
            max_vals = max_val
        return (latlon - min_vals) / (max_vals - min_vals)
    
    def minmax_norm_per_sample(self, 
                               lat: float, 
                               lon: float, 
                               min_lat: float, 
                               max_lat: float, 
                               min_lon: float, 
                               max_lon: float) -> Tuple[float, float]:
        """Perform min-max normalization 
        on latitude and longitude data."""
        return (lat - min_lat) / (max_lat - min_lat), (lon - min_lon) / (max_lon - min_lon)
    
    def lat_lon_height_to_xyz(self, lat: float, lon: float, altitude: float=0) -> Tuple[float, float, float]:
        a=6378137.0
        e_squared=0.00669437999
        latr=np.deg2rad(lat)
        lonr=np.deg2rad(lon)         #longitude in radians
        R = a/np.sqrt(1-e_squared*(np.sin(latr)**2))**0.5
        x=(R+altitude)*np.cos(latr)*np.cos(lonr)
        y=(R+altitude)*np.cos(latr)*np.sin(lonr)
        z=(R+altitude-e_squared*R)*np.sin(latr)
        return x, y, z

    def unitx_vector(self,
                    unitx_latlon_arr: NDArray, 
                    unitx_height_arr: NDArray,
                    ) -> NDArray:
        unitx_latlon_height_arr = np.column_stack((unitx_latlon_arr, unitx_height_arr))

        # Convert latitude/longitude cartesian coordinates
        unitx_xyz = np.array([self.lat_lon_height_to_xyz(lat, lon, height) for lat, lon, height in unitx_latlon_height_arr])
        # normalize the vector
        unitx_xyz = unitx_xyz / np.linalg.norm(unitx_xyz, axis=1, keepdims=True)
        return unitx_xyz
    
    def unit2_to_unit1_vector(self,
                               unit1_latlon_arr: NDArray, 
                               unit2_latlon_arr: NDArray,
                               unit1_height_arr: NDArray,
                               unit2_height_arr: NDArray) -> NDArray:
        """Calculate distance and angle between two sets of coordinates.
        [BS]unit1_height is the BS vertical distance 
        from the ground in meters (we set to 0).
        
        [UE]unit2_height is the drone vertical distance 
        from the ground in meters (we use the value from the dataset).
        """
        unit1_latlon_height_arr = np.column_stack((unit1_latlon_arr, unit1_height_arr))
        unit2_latlon_height_arr = np.column_stack((unit2_latlon_arr, unit2_height_arr))

        # Convert latitude/longitude cartesian coordinates
        unit1_xyz = np.array([self.lat_lon_height_to_xyz(lat, lon, height) for lat, lon, height in unit1_latlon_height_arr])
        unit2_xyz = np.array([self.lat_lon_height_to_xyz(lat, lon, height) for lat, lon, height in unit2_latlon_height_arr])

        # subtract the xyz coordinates -> get new coordinates
        dx = unit2_xyz[:, 0] - unit1_xyz[:, 0]
        dy = unit2_xyz[:, 1] - unit1_xyz[:, 1]
        dz = unit2_xyz[:, 2] - unit1_xyz[:, 2]
        r = np.sqrt(dx**2 + dy**2 + dz**2)

        # normalize the vector
        dx = dx / r
        dy = dy / r
        dz = dz / r

        return np.column_stack((dx, dy, dz))
    
    def unit2_to_unit1_vector_per_sample(self,
                                         unit1_lat: float,
                                         unit1_lon: float,
                                         unit2_lat: float,
                                         unit2_lon: float,
                                         unit1_height: float,
                                         unit2_height: float) -> Tuple[float, float, float]:
        """Calculate normalized direction vector from unit1 to unit2 for a single sample.
        
        Args:
            unit1_lat (float): Latitude of unit1 (e.g., BS)
            unit1_lon (float): Longitude of unit1
            unit2_lat (float): Latitude of unit2 (e.g., UE/drone)
            unit2_lon (float): Longitude of unit2
            unit1_height (float): Height of unit1 from ground in meters
            unit2_height (float): Height of unit2 from ground in meters
            
        Returns:
            Tuple[float, float, float]: Normalized direction vector (dx, dy, dz)
        """
        # Convert latitude/longitude to cartesian coordinates
        unit1_x, unit1_y, unit1_z = self.lat_lon_height_to_xyz(unit1_lat, unit1_lon, unit1_height)
        unit2_x, unit2_y, unit2_z = self.lat_lon_height_to_xyz(unit2_lat, unit2_lon, unit2_height)
        
        # Calculate vector components
        dx = unit2_x - unit1_x
        dy = unit2_y - unit1_y
        dz = unit2_z - unit1_z
        
        # Calculate distance
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Normalize the vector
        dx_norm = dx / r
        dy_norm = dy / r
        dz_norm = dz / r
        
        return dx_norm, dy_norm, dz_norm
    