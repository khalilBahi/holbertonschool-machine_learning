#!/usr/bin/env python3

def matrix_transpose(matrix):
    return [list(row) for row in zip(*matrix)]
