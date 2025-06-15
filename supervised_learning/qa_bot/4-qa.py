#!/usr/bin/env python3
"""
4. Multi-reference Question Answering
"""

semantic_search = __import__('3-semantic_search').semantic_search
question_answer_single = __import__('0-qa').question_answer


def question_answer(corpus_path):
    """
    Answers questions from multiple reference texts in an interactive loop

    Args:
        corpus_path (str): Path to the corpus of reference documents
    """
    while True:
        try:
            # Get user input
            user_input = input("Q: ")

            # Check for exit conditions (case insensitive)
            if user_input.lower() in ['exit', 'quit', 'goodbye', 'bye']:
                print("A: Goodbye")
                break

            # Use semantic search to find the most relevant document
            most_relevant_doc = semantic_search(corpus_path, user_input)

            if not most_relevant_doc:
                print("A: Sorry, I do not understand your question.")
                continue

            # Use the question answering function on the most relevant document
            answer = question_answer_single(user_input, most_relevant_doc)

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
