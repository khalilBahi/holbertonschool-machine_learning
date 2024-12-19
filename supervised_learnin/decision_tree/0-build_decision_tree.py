#!/usr/bin/env python3
""" Task 0: 0. Depth of a decision tree """
import numpy as np


class Node:
    """
    A class representing a node in a decision tree.

    Attributes:
    feature : int or None
        The feature used for splitting the data.
    threshold : float or None
        The threshold value for the split.
    left_child : Node or None
        The left child node.
    right_child : Node or None
        The right child node.
    is_leaf : bool
        Boolean indicating if the node is a leaf.
    is_root : bool
        Boolean indicating if the node is the root.
    sub_population : any
        The subset of data at this node.
    depth : int
        The depth of the node in the tree.

    Methods:
    max_depth_below():
        Calculates the maximum depth of the subtree rooted at this node.
    """

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
        """
        Initializes a Node with the given parameters.

        Parameters:
        feature : int or None, optional
            The feature used for splitting the data (default is None).
        threshold : float or None, optional
            The threshold value for the split (default is None).
        left_child : Node or None, optional
            The left child node (default is None).
        right_child : Node or None, optional
            The right child node (default is None).
        is_root : bool, optional
            Boolean indicating if the node is the root (default is False).
        depth : int, optional
            The depth of the node in the tree (default is 0).
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Calculates the maximum depth of the subtree rooted at this node.

        Returns:
        int
            The maximum depth of the subtree.
        """
        if self.is_leaf:
            return self.depth
        if self.left_child:
            left_depth = self.left_child.max_depth_below()
        else:
            left_depth = self.depth
        if self.right_child:
            right_depth = self.right_child.max_depth_below()
        else:
            right_depth = self.depth
        return max(left_depth, right_depth)


class Leaf(Node):
    """
    A class representing a leaf node in a decision tree, inheriting from Node.

    Attributes:
    value : any
        The value predicted by the leaf.
    depth : int
        The depth of the leaf in the tree.

    Methods:
    max_depth_below():
        Returns the depth of the leaf.
    """

    def __init__(self, value, depth=None):
        """
        Initializes a Leaf with the given parameters.

        Parameters:
        value : any
            The value predicted by the leaf.
        depth : int, optional
            The depth of the leaf in the tree (default is None).
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of the leaf.

        Returns:
        int
            The depth of the leaf.
        """
        return self.depth


class Decision_Tree():
    """
    A class representing a decision tree.

    Attributes:
    rng : numpy.random.Generator
        Random number generator for reproducibility.
    root : Node
        The root node of the tree.
    explanatory : any
        The explanatory features of the dataset.
    target : any
        The target values of the dataset.
    max_depth : int
        The maximum depth of the tree.
    min_pop : int
        The minimum population required to split a node.
    split_criterion : str
        The criterion used to split nodes.
    predict : any
        Method to predict the target value for a given set of features.

    Methods:
    depth():
        Returns the maximum depth of the tree.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initializes a Decision_Tree with the given parameters.

        Parameters:
        max_depth : int, optional
            The maximum depth of the tree (default is 10).
        min_pop : int, optional
            The minimum population required to split a node (default is 1).
        seed : int, optional
            Seed for the random number generator (default is 0).
        split_criterion : str, optional
            The criterion used to split nodes (default is "random").
        root : Node or None, optional
            The root node of the tree (default is None).
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Returns the maximum depth of the tree.

        Returns:
        int
            The maximum depth of the tree.
        """
        return self.root.max_depth_below()
