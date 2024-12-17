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
from scipy.signal import savgol_filter
import os
from dotenv import load_dotenv

class ImprovedRouteAnalyzer:
    def __init__(self, api_key: str):
        """Initialize with Google Maps API key."""
        self.api_key = api_key
        self.gmaps = googlemaps.Client(key=api_key)
        self.MAX_REALISTIC_GRADE = 40.0  # Maximum realistic grade percentage
        
    def _parse_location(self, location: str) -> Union[Tuple[float, float], str]:
        """Parse location string to coordinates or address."""
        coord_pattern = r'^-?\d+\.?\d*,-?\d+\.?\d*$'
        if re.match(coord_pattern, location):
            lat, lng = map(float, location.split(','))
            return (lat, lng)
        return location

    def get_route_points(self, origin: str, destination: str, sampling_interval: float = 50.0) -> List[Dict]:
        """
        Get route points sampled at specified intervals.
        
        Args:
            origin: Starting point (coordinates or address)
            destination: End point (coordinates or address)
            sampling_interval: Distance between points in meters
            
        Returns:
            List of dicts containing lat, lng points
        """
        # Parse locations
        origin_loc = self._parse_location(origin)
        dest_loc = self._parse_location(destination)
        
        # Get directions
        directions = self.gmaps.directions(
            origin_loc,
            dest_loc,
            mode="driving",
            departure_time=datetime.now(),
            alternatives=True
        )
        
        if not directions:
            raise ValueError("No route found between the specified coordinates")
        
        # Find the route with toll roads
        toll_route = None
        for route in directions:
            for leg in route['legs']:
                if any('toll road' in step.get('html_instructions', '').lower() for step in leg['steps']):
                    toll_route = route
                    break
            if toll_route:
                break
        
        # If no toll route found, use the first route
        route = toll_route if toll_route else directions[0]
        
        # Store route information
        self.route_info = {
            'overview_polyline': route['overview_polyline']['points'],
            'duration': route['legs'][0]['duration']['text'],
            'duration_in_traffic': route['legs'][0].get('duration_in_traffic', {}).get('text', 'N/A'),
            'distance': route['legs'][0]['distance']['text'],
            'start_address': route['legs'][0]['start_address'],
            'end_address': route['legs'][0]['end_address'],
            'steps': [],
            'has_toll': False
        }
        
        # Extract steps information
        for step in route['legs'][0]['steps']:
            instructions = re.sub('<[^<]+?>', '', step['html_instructions'])
            is_toll = 'toll road' in instructions.lower()
            if is_toll:
                self.route_info['has_toll'] = True
            
            self.route_info['steps'].append({
                'distance': step['distance']['text'],
                'duration': step['duration']['text'],
                'instructions': instructions,
                'is_toll': is_toll
            })
        
        # Extract polyline points
        polyline_points = []
        for leg in route['legs']:
            for step in leg['steps']:
                points = polyline.decode(step['polyline']['points'])
                polyline_points.extend(points)
                
        # Resample points at regular intervals
        resampled_points = []
        current_distance = 0
        last_point = polyline_points[0]
        resampled_points.append({
            'lat': last_point[0],
            'lng': last_point[1],
            'distance': 0
        })
        
        for point in polyline_points[1:]:
            segment_distance = geodesic(last_point, point).meters
            while current_distance + segment_distance >= sampling_interval:
                # Interpolate new point
                fraction = (sampling_interval - current_distance) / segment_distance
                new_lat = last_point[0] + fraction * (point[0] - last_point[0])
                new_lng = last_point[1] + fraction * (point[1] - last_point[1])
                current_distance = 0
                last_point = (new_lat, new_lng)
                resampled_points.append({
                    'lat': new_lat,
                    'lng': new_lng,
                    'distance': len(resampled_points) * sampling_interval
                })
                segment_distance = geodesic(last_point, point).meters
            
            current_distance += segment_distance
            last_point = point
            
        return resampled_points
        
    def _smooth_elevation_data(self, elevations: np.array, window_length: int = 9) -> np.array:
        """
        Smooth elevation data using Savitzky-Golay filter.
        
        Args:
            elevations: Array of elevation values
            window_length: Length of the smoothing window (odd number)
        
        Returns:
            Smoothed elevation array
        """
        if len(elevations) < window_length:
            return elevations
            
        # Ensure window length is odd
        if window_length % 2 == 0:
            window_length += 1
            
        # Apply Savitzky-Golay filter
        try:
            smoothed = savgol_filter(elevations, window_length, 3)
            return smoothed
        except ValueError:
            # If error occurs (e.g., window too large), fall back to simple moving average
            return pd.Series(elevations).rolling(window=5, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
            
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
        
    def analyze_route(self, origin: str, destination: str, base_sampling_interval: float = 50.0) -> pd.DataFrame:
        """
        Analyze a route's elevation profile with improved accuracy.
        
        Args:
            origin: Starting point
            destination: End point
            base_sampling_interval: Base distance between points in meters (default: 50m)
            
        Returns:
            DataFrame with distance, elevation, grade data
        """
        # Get route points with basic sampling
        points = self.get_route_points(origin, destination, base_sampling_interval)
        
        # Get elevations
        points = self.get_elevations(points)
        
        # Convert to DataFrame
        df = pd.DataFrame(points)
        
        # Smooth elevation data
        df['elevation_raw'] = df['elevation'].copy()  # Keep raw data
        df['elevation'] = self._smooth_elevation_data(df['elevation'].values)
        
        # Calculate grades
        df['elevation_change'] = df['elevation'].diff()
        df['distance_change'] = df['distance'].diff()
        df['grade'] = (df['elevation_change'] / df['distance_change']) * 100
        
        # Clip unrealistic grades
        df['grade'] = df['grade'].clip(lower=-self.MAX_REALISTIC_GRADE, upper=self.MAX_REALISTIC_GRADE)
        
        # Calculate cumulative elevation changes
        df['elevation_gain'] = df['elevation_change'].clip(lower=0).cumsum()
        df['elevation_loss'] = df['elevation_change'].clip(upper=0).cumsum()
        
        # Add location names for significant points
        significant_points = [
            (df.iloc[0], 'Start'),
            (df.iloc[-1], 'End')
        ]
        
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
    
    def plot_route_map(self, df: pd.DataFrame, grade_threshold: float = 15.0):
        """
        Generate and save a static map with the route, highlighting steep segments.
        
        Args:
            df: DataFrame with route data
            grade_threshold: Grade percentage to mark as steep
        """
        try:
            # Find steep sections
            steep_ascents = df[df['grade'] > grade_threshold]
            steep_descents = df[df['grade'] < -grade_threshold]
            
            # Base map URL
            map_url = f"https://maps.googleapis.com/maps/api/staticmap?"
            map_url += f"size=2048x2048&scale=2"  # High resolution map
            map_url += f"&maptype=roadmap"
            
            # Main route path (blue)
            map_url += f"&path=color:blue|weight:2|enc:{self.route_info['overview_polyline']}"
            
            # Add steep ascents (red)
            for _, section in steep_ascents.iterrows():
                map_url += f"&markers=color:red|size:tiny|{section['lat']},{section['lng']}"
                
            # Add steep descents (orange)
            for _, section in steep_descents.iterrows():
                map_url += f"&markers=color:orange|size:tiny|{section['lat']},{section['lng']}"
            
            # Start and end markers
            map_url += f"&markers=color:green|label:A|{df.iloc[0]['lat']},{df.iloc[0]['lng']}"
            map_url += f"&markers=color:red|label:B|{df.iloc[-1]['lat']},{df.iloc[-1]['lng']}"
            map_url += f"&key={self.api_key}"
            
            # Download and save the map
            response = requests.get(map_url)
            if response.status_code == 200:
                with open("route_map.png", "wb") as f:
                    f.write(response.content)
                print("Route map saved as 'route_map.png'")
                
                # Display the map using matplotlib
                img = plt.imread("route_map.png")
                plt.figure(figsize=(15, 15))
                plt.imshow(img)
                plt.axis('off')
                plt.show()
            else:
                print(f"Failed to generate Google Maps static map. Status code: {response.status_code}")
                print("Response content:", response.text[:500])  # Print first 500 chars of error message
        except Exception as e:
            print(f"Error generating route map: {str(e)}")

    def plot_elevation_profile(self, df: pd.DataFrame, title: str = "Route Elevation Profile"):
        """Plot elevation profile with improved visualization."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12), height_ratios=[3, 1])
        
        # Plot both raw and smoothed elevation profiles
        ax1.plot(df['distance'], df['elevation_raw'], 'lightgray', alpha=0.5, label='Raw Elevation')
        ax1.plot(df['distance'], df['elevation'], 'b-', linewidth=2, label='Smoothed Elevation')
        
        ax1.set_xlabel('Distance (m)')
        ax1.set_ylabel('Elevation (m)')
        ax1.grid(True)
        ax1.legend()
        
        if 'location_names' in df.attrs:
            title = title + "\n" + " | ".join(df.attrs['location_names'])
        ax1.set_title(title)
        
        # Grade profile with realistic limits highlighted
        ax2.plot(df['distance'], df['grade'], 'r-', linewidth=1)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        ax2.axhspan(15, self.MAX_REALISTIC_GRADE, color='yellow', alpha=0.2, label='Steep')
        ax2.axhspan(-self.MAX_REALISTIC_GRADE, -15, color='yellow', alpha=0.2)
        
        ax2.set_xlabel('Distance (m)')
        ax2.set_ylabel('Grade (%)')
        ax2.set_ylim(-self.MAX_REALISTIC_GRADE * 1.1, self.MAX_REALISTIC_GRADE * 1.1)
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def print_route_info(self):
        """Print detailed information about the route."""
        print("\nRoute Information:")
        print("=" * 50)
        print(f"From: {self.route_info['start_address']}")
        print(f"To: {self.route_info['end_address']}")
        print(f"Total Distance: {self.route_info['distance']}")
        print(f"Normal Duration: {self.route_info['duration']}")
        print(f"Duration in Traffic: {self.route_info['duration_in_traffic']}")
        print(f"Route includes toll roads: {'Yes' if self.route_info['has_toll'] else 'No'}")
        
        print("\nDetailed Steps:")
        print("=" * 50)
        for i, step in enumerate(self.route_info['steps'], 1):
            if step['is_toll']:  # Only show toll road steps
                print(f"\nStep {i}:")
                print(f"  • {step['instructions']} [TOLL ROAD]")
                print(f"  • Distance: {step['distance']}")
                print(f"  • Duration: {step['duration']}")

def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    if not api_key:
        raise ValueError("Google Maps API key not found in environment variables")
    
    # Example usage
    analyzer = ImprovedRouteAnalyzer(api_key)
    
    # Analyze route using coordinates
    df = analyzer.analyze_route(
        origin="-7.076283,110.4264041",      # Start point
        destination="-7.6694298,111.1413488", # End point
        base_sampling_interval=25.0
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
    
    # Find steep sections (>15% grade)
    steep_sections = df[abs(df['grade']) > 15]
    if not steep_sections.empty:
        print("\nSteep sections (>15% grade):")
        for _, section in steep_sections.iterrows():
            print(f"Distance {section['distance']:.0f}m: {section['grade']:.1f}% grade")
    
    # Print route information
    analyzer.print_route_info()
    
    # Plot elevation profile
    fig = analyzer.plot_elevation_profile(df)
    
    # Save figure
    fig.savefig('route_elevation_profile.png', dpi=600, bbox_inches='tight')
    plt.show()

    # Generate and display route map
    analyzer.plot_route_map(df)

#%% main call
if __name__ == "__main__":
    main()

#%%