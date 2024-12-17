# Modified version of RouteElevationAnalyzer that uses Routes API
import googlemaps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from geopy.distance import geodesic
from typing import List, Tuple, Dict, Union
import re
import polyline
import requests
from google.maps import routing_v2
from google.maps.routing_v2 import RoutesClient
from google.maps.routing_v2.types import (
    ComputeRoutesRequest, 
    Waypoint, 
    RouteTravelMode, 
    RoutingPreference, 
    ComputeRoutesResponse, 
    PolylineQuality
)

class RouteElevationAnalyzerV2:
    def __init__(self, api_key: str):
        """Initialize with Google Maps API key."""
        self.api_key = api_key  # Store the API key
        self.gmaps = googlemaps.Client(key=api_key)
        self.routes_client = RoutesClient(client_options={"api_key": api_key})
        
    def _parse_location(self, location: str) -> Union[Tuple[float, float], str]:
        """
        Parse location string to either coordinates or address.
        Accepts formats:
        - "latitude,longitude" (e.g. "-6.9175,107.6191")
        - Address string (e.g. "Bandung, Indonesia")
        """
        coord_pattern = r'^-?\d+\.?\d*,-?\d+\.?\d*$'
        if re.match(coord_pattern, location):
            lat, lng = map(float, location.split(','))
            return (lat, lng)
        return location
        
    def _get_route_data(self, origin: str, destination: str) -> Dict:
        """Get detailed route information using Routes API."""
        try:
            # Parse locations
            origin_loc = self._parse_location(origin)
            dest_loc = self._parse_location(destination)
            
            print(f"\nDebug - Parsed locations - Origin: {origin_loc}, Destination: {dest_loc}")
            
            # Create waypoints
            if isinstance(origin_loc, tuple):
                origin_point = Waypoint(
                    location={
                        "lat_lng": {
                            "latitude": origin_loc[0],
                            "longitude": origin_loc[1]
                        }
                    }
                )
            else:
                origin_point = Waypoint(address=origin_loc)
                
            if isinstance(dest_loc, tuple):
                dest_point = Waypoint(
                    location={
                        "lat_lng": {
                            "latitude": dest_loc[0],
                            "longitude": dest_loc[1]
                        }
                    }
                )
            else:
                dest_point = Waypoint(address=dest_loc)
            
            # Create request
            request = ComputeRoutesRequest(
                origin=origin_point,
                destination=dest_point,
                travel_mode=RouteTravelMode.DRIVE,
                routing_preference=RoutingPreference.TRAFFIC_AWARE,
                compute_alternative_routes=False,
                polyline_quality=PolylineQuality.HIGH_QUALITY,
                language_code="id",
                extra_computations=["TOLLS"],
                route_modifiers={"avoid_tolls": False}  # Explicitly allow toll roads
            )
            
            # Define the field mask as a header
            headers = {
                'X-Goog-FieldMask': 'routes.distanceMeters,routes.duration,routes.polyline.encodedPolyline,routes.travelAdvisory.tollInfo'
            }
            
            print("\nDebug - Sending request to Routes API...")
            try:
                response = self.routes_client.compute_routes(
                    request=request,
                    metadata=[('x-goog-fieldmask', headers['X-Goog-FieldMask'])]
                )
                print("\nDebug - Got response from Routes API")
                print("\nDebug - Response type:", type(response))
                print("\nDebug - Response content:", response)
            except Exception as api_error:
                print(f"\nDebug - Routes API error: {type(api_error)} - {str(api_error)}")
                if hasattr(api_error, 'details'):
                    print(f"\nDebug - API Error details: {api_error.details}")
                return {
                    'error': f"Routes API error: {str(api_error)}",
                    'polyline': None,
                    'distance': None,
                    'duration': None,
                    'toll_info': {
                        'has_tolls': False,
                        'toll_passes': [],
                        'estimated_price': None,
                        'currency_code': None
                    }
                }
            
            # Rest of the method remains the same...
            route_data = {
                'polyline': None,
                'distance': None,
                'duration': None,
                'toll_info': {
                    'has_tolls': False,
                    'toll_passes': [],
                    'estimated_price': None,
                    'currency_code': None
                }
            }
            
            if not hasattr(response, 'routes') or not response.routes:
                print("\nDebug - No routes found in response")
                return route_data
            
            print(f"\nDebug - Found {len(response.routes)} routes")
            route = response.routes[0]
            
            # Extract basic route data
            if hasattr(route, 'polyline') and hasattr(route.polyline, 'encoded_polyline'):
                route_data['polyline'] = route.polyline.encoded_polyline
            if hasattr(route, 'distance_meters'):
                route_data['distance'] = route.distance_meters
            if hasattr(route, 'duration'):
                route_data['duration'] = route.duration.seconds
                
            print("\nDebug - Basic route data extracted:", {k:v for k,v in route_data.items() if k != 'polyline'})
            
            # Extract toll information if available
            if hasattr(route, 'travel_advisory') and route.travel_advisory:
                if hasattr(route.travel_advisory, 'toll_info') and route.travel_advisory.toll_info:
                    toll_info = route.travel_advisory.toll_info
                    route_data['toll_info']['has_tolls'] = True
                    
                    if hasattr(toll_info, 'estimated_price'):
                        price = toll_info.estimated_price
                        route_data['toll_info'].update({
                            'estimated_price': price.units + price.nanos / 1e9,
                            'currency_code': price.currency_code
                        })
                    
                    if hasattr(toll_info, 'toll_passes'):
                        route_data['toll_info']['toll_passes'] = list(toll_info.toll_passes)
            
            print("\nDebug - Toll info extracted:", route_data['toll_info'])
            return route_data
            
        except Exception as e:
            print(f"\nDebug - Error in _get_route_data: {type(e)} - {str(e)}")
            if hasattr(e, 'details'):
                print(f"\nDebug - Error details: {e.details}")
            if hasattr(e, '__dict__'):
                print(f"\nDebug - Error dict: {e.__dict__}")
            return None
        
    def get_route_points(self, origin: str, destination: str, sampling_interval: float = 50.0) -> List[Dict]:
        """Get route points sampled at specified intervals."""
        # Get route data from Routes API
        route_data = self._get_route_data(origin, destination)
        print("\nDebug - Route Data received:", route_data)  # Debug print
        
        if not route_data:
            raise ValueError("Failed to get route data from Routes API")
        if not route_data['polyline']:
            if route_data.get('error'):
                raise ValueError(f"API Error: {route_data['error']}")
            raise ValueError(f"No polyline found in route data. Full response: {route_data}")
            
        # Store route information
        self.route_info = {
            'total_distance': f"{route_data['distance']/1000:.1f} km",
            'total_duration': f"{route_data['duration']/3600:.1f} hours",
            'has_tolls': route_data['toll_info']['has_tolls'],
            'estimated_toll_price': route_data['toll_info']['estimated_price'],
            'toll_currency': route_data['toll_info']['currency_code'],
            'toll_passes': route_data['toll_info']['toll_passes'],
            'overview_polyline': route_data['polyline']
        }
        
        # Debug print route info
        print("\nDebug - Route Info created:", self.route_info)
        
        # Decode polyline
        try:
            polyline_points = polyline.decode(route_data['polyline'])
            print(f"\nDebug - Decoded {len(polyline_points)} points from polyline")
        except Exception as e:
            print(f"\nDebug - Error decoding polyline: {str(e)}")
            raise
            
        # Rest of the method remains the same...
        resampled_points = []
        current_distance = 0
        last_point = polyline_points[0]
        resampled_points.append({
            'lat': last_point[0],
            'lng': last_point[1],
            'distance': 0
        })
    
    def get_elevations(self, points: List[Dict]) -> List[Dict]:
        """Get elevations for a list of points."""
        # Batch points into groups of 512 (API limit)
        point_groups = [points[i:i + 512] for i in range(0, len(points), 512)]
        
        for group in point_groups:
            # Get elevations for this batch
            locations = [(p['lat'], p['lng']) for p in group]
            results = self.gmaps.elevation(locations)
            
            # Add elevation data to points
            for point, result in zip(group, results):
                point['elevation'] = result['elevation']
                
        return points
    
    def analyze_route(self, origin: str, destination: str, sampling_interval: float = 50.0) -> pd.DataFrame:
        """Analyze a route's elevation profile."""
        # Get route points
        points = self.get_route_points(origin, destination, sampling_interval)
        
        # Get elevations
        points = self.get_elevations(points)
        
        # Convert to DataFrame
        df = pd.DataFrame(points)
        
        # Calculate grades with smoothing
        window_size = 5  # Points to use for smoothing (adjust as needed)
        df['elevation_smooth'] = df['elevation'].rolling(window=window_size, center=True).mean()
        df['elevation_change'] = df['elevation_smooth'].diff()
        df['grade'] = (df['elevation_change'] / sampling_interval) * 100
        
        # Calculate cumulative elevation gain/loss
        df['elevation_gain'] = df['elevation_change'].clip(lower=0).cumsum()
        df['elevation_loss'] = df['elevation_change'].clip(upper=0).cumsum()
        
        # Add reverse geocoding for significant points
        significant_points = [
            (df.iloc[0], 'Start'),
            (df.iloc[-1], 'End')
        ]
        
        # Add location names for significant points
        location_names = []
        for point, point_type in significant_points:
            try:
                reverse_geocode = self.gmaps.reverse_geocode((point['lat'], point['lng']))
                if reverse_geocode:
                    location_names.append(f"{point_type}: {reverse_geocode[0]['formatted_address']}")
            except Exception as e:
                location_names.append(f"{point_type}: {point['lat']:.6f}, {point['lng']:.6f}")
        
        df.attrs['location_names'] = location_names
        
        return df
    
    def plot_elevation_profile(self, df: pd.DataFrame, title: str = "Route Elevation Profile"):
        """Plot elevation profile with grades."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12), height_ratios=[3, 1])
        
        # Elevation profile
        ax1.plot(df['distance'], df['elevation'], 'b-', linewidth=2, alpha=0.5, label='Raw')
        ax1.plot(df['distance'], df['elevation_smooth'], 'r-', linewidth=2, label='Smoothed')
        ax1.set_xlabel('Distance (m)')
        ax1.set_ylabel('Elevation (m)')
        ax1.grid(True)
        ax1.legend()
        
        # Add location names and route info if available
        title_parts = [title]
        if 'location_names' in df.attrs:
            title_parts.append(" | ".join(df.attrs['location_names']))
        if hasattr(self, 'route_info'):
            info_parts = []
            info_parts.append(f"Distance: {self.route_info['total_distance']}")
            info_parts.append(f"Duration: {self.route_info['total_duration']}")
            if self.route_info['has_tolls']:
                toll_info = f"Toll: {self.route_info['estimated_toll_price']} {self.route_info['toll_currency']}"
                info_parts.append(toll_info)
            title_parts.append(" | ".join(info_parts))
        
        ax1.set_title('\n'.join(title_parts))
        
        # Grade profile
        ax2.plot(df['distance'], df['grade'], 'r-', linewidth=1)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        ax2.set_xlabel('Distance (m)')
        ax2.set_ylabel('Grade (%)')
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_route_map(self, df: pd.DataFrame, zoom: int = 9):
        """Generate a static map with the route."""
        try:
            # First check if we can use static maps
            test_url = f"https://maps.googleapis.com/maps/api/staticmap?center=0,0&zoom=1&size=100x100&key={self.api_key}"
            test_response = requests.get(test_url)
            
            if test_response.status_code == 403:
                print("Error: Google Maps Static API access forbidden (403).")
                print("Please make sure Static Maps API is enabled")
                print("\nFalling back to coordinate plot...")
                
                plt.figure(figsize=(12, 8))
                plt.plot(df['lng'], df['lat'], 'b-', linewidth=2)
                plt.plot(df.iloc[0]['lng'], df.iloc[0]['lat'], 'go', label='Start', markersize=10)
                plt.plot(df.iloc[-1]['lng'], df.iloc[-1]['lat'], 'ro', label='End', markersize=10)
                plt.title('Route Map (Coordinate View)')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.grid(True)
                plt.legend()
                plt.savefig('route_map.png', dpi=300, bbox_inches='tight')
                plt.close()
                return
            
            # Generate static map URL using overview_polyline
            map_url = f"https://maps.googleapis.com/maps/api/staticmap?"
            map_url += f"size=2048x2048&scale=2"  # High resolution map
            map_url += f"&maptype=roadmap"
            map_url += f"&path=enc:{self.route_info['overview_polyline']}"
            map_url += f"&markers=color:green|label:A|{df.iloc[0]['lat']},{df.iloc[0]['lng']}"
            map_url += f"&markers=color:red|label:B|{df.iloc[-1]['lat']},{df.iloc[-1]['lng']}"
            map_url += f"&key={self.api_key}"
            
            response = requests.get(map_url)
            if response.status_code == 200:
                with open("route_map.png", "wb") as f:
                    f.write(response.content)
                print("Route map saved as 'route_map.png'")
            else:
                print(f"Failed to generate static map. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error generating route map: {str(e)}")
    
    def print_route_info(self):
        """Print detailed route information including toll segments."""
        print("\nRoute Information:")
        print(f"Total Distance: {self.route_info['total_distance']}")
        print(f"Total Duration: {self.route_info['total_duration']}")
        
        if self.route_info['has_tolls']:
            print("\nToll Information:")
            if self.route_info['estimated_toll_price'] is not None:
                print(f"Estimated Toll Cost: {self.route_info['estimated_toll_price']} {self.route_info['toll_currency']}")
            if self.route_info['toll_passes']:
                print("Required Toll Passes:")
                for toll_pass in self.route_info['toll_passes']:
                    print(f"- {toll_pass}")
        else:
            print("\nNo toll roads detected in this route")

def main():
    # Example usage
    api_key = "AIzaSyDFlIAWpmwrpQdYPMTQLSkTVjohoeIiOws"
    analyzer = RouteElevationAnalyzerV2(api_key)
    
    # Example: Analyze route using coordinates
    df = analyzer.analyze_route(
        origin="-7.076283,110.4264041",      # AZ Consulting
        destination="-7.6694298,111.1413488",  # Villa Avocado
        sampling_interval=50.0
    )
    
    # Print summary statistics
    print("\nRoute Analysis:")
    print(f"Total distance: {df['distance'].max():.1f}m")
    print(f"Starting elevation: {df['elevation'].iloc[0]:.1f}m")
    print(f"Ending elevation: {df['elevation'].iloc[-1]:.1f}m")
    print(f"Total elevation gain: {df['elevation_gain'].max():.1f}m")
    print(f"Total elevation loss: {abs(df['elevation_loss'].min()):.1f}m")
    print(f"Maximum grade: {df['grade'].max():.1f}%")
    print(f"Minimum grade: {df['grade'].min():.1f}%")
    
    # Print location names
    if 'location_names' in df.attrs:
        print("\nRoute Details:")
        for name in df.attrs['location_names']:
            print(name)
    
    # Find steep sections (>10% grade)
    steep_sections = df[abs(df['grade']) > 10]
    if not steep_sections.empty:
        print("\nSteep sections (>10% grade):")
        for _, section in steep_sections.iterrows():
            print(f"Distance {section['distance']:.0f}m: {section['grade']:.1f}% grade")
    
    # Plot elevation profile
    fig = analyzer.plot_elevation_profile(df, "Route Elevation Profile")
    fig.savefig("route_elevation_profile.png", dpi=600)
    plt.show()

    # Print route information including tolls
    analyzer.print_route_info()
    
    # Generate static map
    analyzer.plot_route_map(df)

#%% sample call
if __name__ == "__main__":
    main()

#%%