#!/usr/bin/env python3
""" Task 6: 6. Stacking Bars """
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Generate and display a stacked bar chart of fruit quantities per person.

    Args:
        None

    Returns:
        None
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))

    people = ["Farrah", "Fred", "Felicia"]
    fruit_labels = ["apples", "bananas", "oranges", "peaches"]
    colors = ["red", "yellow", "#ff8000", "#ffe5b4"]

    bar_width = 0.5

    # Initialize the bottom for stacking
    bottom = np.zeros(fruit.shape[1])

    for i, (fruit_count, color, label) in enumerate(
            zip(fruit, colors, fruit_labels)
            ):
        plt.bar(
            people,
            fruit_count,
            width=bar_width,
            color=color,
            label=label,
            bottom=bottom,
        )
        bottom += fruit_count

    plt.ylabel("Quantity of Fruit")
    plt.title("Number of Fruit per Person")
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.legend()

    # Show the plot
    plt.show()
