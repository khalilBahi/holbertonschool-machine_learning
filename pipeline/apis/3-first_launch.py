#!/usr/bin/env python3
"""
Script that displays the first SpaceX launch with detailed information
using the SpaceX API
"""
import requests


def get_first_launch():
    """
    Get the first SpaceX launch and display its information
    
    Format: <launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)
    """
    try:
        # Get all launches
        launches_response = requests.get('https://api.spacexdata.com/v4/launches/upcoming')
        if launches_response.status_code != 200:
            print(f"Error fetching launches: {launches_response.status_code}")
            return
        
        launches = launches_response.json()
        
        # Sort by date_unix to find the first launch
        # If two launches have the same date, use the first one in the API result
        launches_sorted = sorted(launches, key=lambda x: (x['date_unix'], launches.index(x)))
        
        if not launches_sorted:
            print("No launches found")
            return
        
        first_launch = launches_sorted[0]
        
        # Get rocket information
        rocket_id = first_launch['rocket']
        rocket_response = requests.get(f'https://api.spacexdata.com/v4/rockets/{rocket_id}')
        if rocket_response.status_code != 200:
            print(f"Error fetching rocket info: {rocket_response.status_code}")
            return
        
        rocket = rocket_response.json()
        
        # Get launchpad information
        launchpad_id = first_launch['launchpad']
        launchpad_response = requests.get(f'https://api.spacexdata.com/v4/launchpads/{launchpad_id}')
        if launchpad_response.status_code != 200:
            print(f"Error fetching launchpad info: {launchpad_response.status_code}")
            return
        
        launchpad = launchpad_response.json()
        
        # Extract the required information
        launch_name = first_launch['name']
        launch_date = first_launch['date_local']  # Use local time as requested
        rocket_name = rocket['name']
        launchpad_name = launchpad['name']
        launchpad_locality = launchpad['locality']
        
        # Format and print the result
        # Format: <launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)
        print(f"{launch_name} ({launch_date}) {rocket_name} - {launchpad_name} ({launchpad_locality})")
        
    except requests.RequestException as e:
        print(f"Request error: {e}")
    except KeyError as e:
        print(f"Missing key in API response: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    get_first_launch()
