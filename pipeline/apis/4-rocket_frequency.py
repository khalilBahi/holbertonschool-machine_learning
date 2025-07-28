#!/usr/bin/env python3
"""
Script that displays the number of launches per rocket
using the SpaceX API
"""
import requests


def get_rocket_frequency():
    """
    Get the number of launches per rocket and display them
    
    Format: <rocket name>: <number of launches>
    Ordered by number of launches (descending), then alphabetically
    """
    try:
        # Get all launches
        launches_response = requests.get('https://api.spacexdata.com/v4/launches/')
        if launches_response.status_code != 200:
            print(f"Error fetching launches: {launches_response.status_code}")
            return
        
        launches = launches_response.json()
        
        # Get all rockets to map IDs to names
        rockets_response = requests.get('https://api.spacexdata.com/v4/rockets/')
        if rockets_response.status_code != 200:
            print(f"Error fetching rockets: {rockets_response.status_code}")
            return
        
        rockets = rockets_response.json()
        
        # Create a mapping from rocket ID to rocket name
        rocket_id_to_name = {rocket['id']: rocket['name'] for rocket in rockets}
        
        # Count launches per rocket
        rocket_launch_count = {}
        
        for launch in launches:
            rocket_id = launch['rocket']
            rocket_name = rocket_id_to_name.get(rocket_id, 'Unknown')
            
            if rocket_name in rocket_launch_count:
                rocket_launch_count[rocket_name] += 1
            else:
                rocket_launch_count[rocket_name] = 1
        
        # Sort by number of launches (descending), then by name (ascending)
        sorted_rockets = sorted(
            rocket_launch_count.items(),
            key=lambda x: (-x[1], x[0])  # -x[1] for descending count, x[0] for ascending name
        )
        
        # Display the results
        for rocket_name, count in sorted_rockets:
            print(f"{rocket_name}: {count}")
        
    except requests.RequestException as e:
        print(f"Request error: {e}")
    except KeyError as e:
        print(f"Missing key in API response: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    get_rocket_frequency()
