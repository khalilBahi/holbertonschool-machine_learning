#!/usr/bin/env python3
"""
2. Answer Questions
"""

question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    Answers questions from a reference text in an interactive loop

    Args:
        reference (str): The reference text to answer questions from
    """
    while True:
        try:
            # Get user input
            user_input = input("Q: ")

            # Check for exit conditions (case insensitive)
            if user_input.lower() in ['exit', 'quit', 'goodbye', 'bye']:
                print("A: Goodbye")
                break

            # Get answer using the question_answer function
            answer = question_answer(user_input, reference)

            # Print the answer or default message
            if answer:
                print(f"A: {answer}")
            else:
                print("A: Sorry, I do not understand your question.")

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nA: Goodbye")
            break
        except EOFError:
            # Handle EOF (Ctrl+D) gracefully
            print("\nA: Goodbye")
            break
