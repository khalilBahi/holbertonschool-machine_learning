#!/usr/bin/env python3
"""
Script that prints the location of a specific GitHub user using the GitHub API
"""
import requests
import sys
import time


def get_user_location(api_url):
    """
    Get the location of a GitHub user from the API

    Args:
        api_url (str): The full GitHub API URL for the user

    Returns:
        None: Prints the result directly
    """
    try:
        response = requests.get(api_url)

        if response.status_code == 200:
            # User found, get location
            user_data = response.json()
            location = user_data.get("location")

            if location:
                print(location)
            else:
                # If location is None or empty, print None
                print("None")

        elif response.status_code == 404:
            # User not found
            print("Not found")

        elif response.status_code == 403:
            # Rate limited - check for X-Ratelimit-Reset header
            reset_time = response.headers.get("X-Ratelimit-Reset")

            if reset_time:
                # Convert reset time from epoch seconds to minutes from now
                current_time = int(time.time())
                reset_timestamp = int(reset_time)
                minutes_until_reset = max
                (0, (reset_timestamp - current_time) // 60)
                print(f"Reset in {minutes_until_reset} min")
            else:
                # Fallback if no reset header
                print("Reset in 0 min")

        else:
            # Other HTTP errors
            print(f"Error: HTTP {response.status_code}")

    except requests.RequestException as e:
        print(f"Request error: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <GitHub_API_URL>")
        sys.exit(1)

    api_url = sys.argv[1]
    get_user_location(api_url)
