#!/usr/bin/env python3
"""
Module that contains a method to get available ships
that can hold a given number of passengers using SWAPI API
"""
import requests


def availableShips(passengerCount):
    """
    Returns the list of ships that can hold a given number of passengers

    Args:
        passengerCount (int): The minimum number
        of passengers the ship should hold

    Returns:
        list: List of ship names that can hold
        at least passengerCount passengers
    """
    ships = []
    url = "https://swapi-api.hbtn.io/api/starships/"

    while url:
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            # Process each ship in the current page
            for ship in data.get("results", []):
                passengers = ship.get("passengers", "0")

                # Handle different passenger count formats
                if passengers == "n/a" or passengers == "unknown":
                    continue

                # Remove commas and convert to int
                try:
                    passengers_clean = passengers.replace(",", "")
                    passengers_int = int(passengers_clean)

                    # Check if ship can hold the required number of passengers
                    if passengers_int >= passengerCount:
                        ships.append(ship.get("name"))
                except (ValueError, AttributeError):
                    # Skip ships with invalid passenger data
                    continue

            # Get next page URL for pagination
            url = data.get("next")

        except requests.RequestException:
            # If there's an error with the request, break the loop
            break

    return ships
