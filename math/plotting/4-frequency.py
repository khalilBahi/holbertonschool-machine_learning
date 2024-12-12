#!/usr/bin/env python3
""" Task 4: 4. Frequency """
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    plot a histogram of student scores for a project.

    Args:
        None

    Returns:
        None
    """

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    bins = range(0, 101, 10)  # Bins every 10 units from 0 to 100
    plt.hist(student_grades, bins=bins, edgecolor="black")

    plt.xlim(0, 100)
    plt.ylim((0, 30))

    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")

    # Show the plot
    plt.show()
