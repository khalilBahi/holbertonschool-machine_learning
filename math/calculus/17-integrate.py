#!/usr/bin/env python3
"""task 17: 17. Integrate"""

def poly_integral(poly, C=0):
    """
    Calculate the integral of a polynomial.

    Args:
        poly (list): List of coefficients representing a polynomial.
        C (int): The integration constant.

    Returns:
        list: A new list of coefficients representing the integral of the polynomial.
        None: If poly or C is not valid.
    """

    if not isinstance(poly, list) or not all(isinstance(c, (int, float)) for c in poly):
        return None

    if not isinstance(C, (int, float)):
        return None

    # Compute the integral: divide each coefficient by its new power index.
    integral = [C] + [coeff / (power + 1) for power, coeff in enumerate(poly)]

    # Convert whole numbers to integers
    integral = [int(term) if isinstance(term, float) and term.is_integer() else term for term in integral]

    return integral
