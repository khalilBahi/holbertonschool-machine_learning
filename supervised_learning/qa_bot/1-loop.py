#!/usr/bin/env python3
"""
1. Create the loop
"""

# Interactive loop
while True:
    try:
        # Get user input
        user_input = input("Q: ")

        # Check for exit conditions (case insensitive)
        if user_input.lower() in ['exit', 'quit', 'goodbye', 'bye']:
            print("A: Goodbye")
            break

        # For now, just print empty answer for any other input
        # This matches the expected behavior from the example
        print("A:")

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nA: Goodbye")
        break
    except EOFError:
        # Handle EOF (Ctrl+D) gracefully
        print("\nA: Goodbye")
        break
