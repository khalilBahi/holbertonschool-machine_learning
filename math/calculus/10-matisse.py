#!/usr/bin/env python3
"""task 10: 10. Derive happiness in oneself from a good day's work"""

def poly_derivative(poly):
    """
    Calculate the derivative of a polynomial.

    Args:
        poly (list): List of coefficients representing a polynomial.

    Returns:
        list: A new list of coefficients representing the derivative of the polynomial.
        None: If poly is not a valid list of coefficients.
    """
    if not isinstance(poly, list) or not all(isinstance(c, (int, float)) for c in poly):
        return None

    # The derivative of a constant polynomial (length 1) is 0.
    if len(poly) <= 1:
        return [0]

    # Compute the derivative: multiply each coefficient by its power index.
    derivative = [coeff * power for power, coeff in enumerate(poly) if power > 0]

    return derivative
