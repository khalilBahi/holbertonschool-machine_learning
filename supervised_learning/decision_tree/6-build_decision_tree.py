#!/usr/bin/env python3
""" Task 1: 1. Number of nodes/leaves in a decision tree"""
import numpy as np


def left_child_add_prefix(text):
    """
    Adds a prefix to each line of the text to
    indicate it is the left child in the tree structure.

    Parameters:
    text : str
        The text to which the prefix will be added.

    Returns:
    str
        The text with the left child prefix added to each line.
    """
    lines = text.split("\n")
    new_text = "    +--" + lines[0] + "\n"
    for x in lines[1:]:
        new_text += ("    |  " + x) + "\n"
    return new_text


def right_child_add_prefix(text):
    """
    Adds a prefix to each line of the text to indicate
    it is the right child in the tree structure.

    Parameters:
    text : str
    The text to which the prefix will be added.
    Returns:
    str
        The text with the right child prefix added to each line.
    """
    lines = text.split("\n")
    new_text = "    +--" + lines[0] + "\n"
    for x in lines[1:]:
        new_text += ("       " + x) + "\n"
    return new_text


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

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the number of nodes in the subtree rooted at this node.

        Parameters:
        only_leaves : bool, optional
            If True, count only the leaf nodes (default is False).

        Returns:
        int
            The number of nodes in the subtree.
        """
        if only_leaves:
            # Count leaves in both children
            return (
                self.left_child.count_nodes_below(only_leaves=True) +
                self.right_child.count_nodes_below(only_leaves=True)
            )
        else:
            # Count all nodes in the subtree
            return (
                1 + self.left_child.count_nodes_below(only_leaves=False) +
                self.right_child.count_nodes_below(only_leaves=False)
            )

    def __str__(self):
        """
        Provides a string representation of the node, including its children.

        Returns:
        str
            A formatted string representing the subtree rooted at this node.
        """
        if self.is_root:
            result = (
                f"root [feature={self.feature}, threshold={self.threshold}]\n"
            )
        else:
            result = (
                f"node [feature={self.feature}, threshold={self.threshold}]\n"
            )

        # Add left child with prefix
        if self.left_child:
            left_str = self.left_child.__str__()
            result += left_child_add_prefix(left_str)

        # Add right child with prefix
        if self.right_child:
            right_str = self.right_child.__str__()
            result += right_child_add_prefix(right_str)

        return result

    def get_leaves_below(self):
        """
        Returns the list of all leaf nodes in the subtree rooted at this node.

        Returns:
        list
            A list of all leaves in the subtree.
        """
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Recursively computes and updates the lower
        and upper bounds dictionaries
        for each node and its children based on the feature thresholds.
        """
        if self.is_root:
            # Initialize bounds at the root
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        # Compute bounds for children
        if self.left_child:
            self.left_child.lower = self.lower.copy()
            self.left_child.upper = self.upper.copy()
            # Update upper bound for the feature
            self.left_child.lower[self.feature] = self.threshold

        if self.right_child:
            self.right_child.lower = self.lower.copy()
            self.right_child.upper = self.upper.copy()
            # Update lower bound for the feature
            self.right_child.upper[self.feature] = self.threshold

        # Recursively update bounds for children
        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()

    def update_indicator(self):
        """
        Compute the indicator function for the current
        node based on the lower and upper bounds.
        """

        def is_large_enough(x):
            """
            Check if each individual has all its features
            greater than the lower bounds.

            Parameters:
            x : np.ndarray
                A 2D NumPy array of shape (n_individuals, n_features).

            Returns:

            np.ndarray
                A 1D NumPy array of boolean values
                indicating if each individual meets the condition.
            """
            return np.all(np.array([x[:, key] > self.lower[key]
                                    for key in self.lower.keys()]), axis=0)

        def is_small_enough(x):
            """
            Check if each individual has all its features
            less than or equal to the upper bounds.

            Parameters:
            x : np.ndarray
                A 2D NumPy array of shape (n_individuals, n_features).

            Returns:
            np.ndarray
                A 1D NumPy array of boolean values indicating
                if each individual meets the condition.
            """
            return np.all(np.array([x[:, key] <= self.upper[key]
                                    for key in self.upper.keys()]), axis=0)

        self.indicator = lambda x: \
            np.all(np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """
        Predict the class for a single individual at the node.

        Parameters:
        x : np.ndarray
            A 1D NumPy array representing the features of a single individual.

        Returns:
        int
            The predicted class for the individual.
        """
        if self.is_leaf:
            return self.value
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


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

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the number of nodes in the subtree rooted at this leaf.

        Parameters:
        only_leaves : bool, optional
            If True, count only the leaf nodes (default is False).

        Returns:
        int
            The number of nodes in the subtree.
        """
        return 1

    def __str__(self):
        """
        Returns a string representation of the leaf node.

        Returns:
        str
            The string representation of the leaf node.
        """
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """
        Returns the list of all leaf nodes in the subtree rooted at this leaf.

        Returns:
        list
            A list containing this leaf.
        """
        return [self]

    def update_bounds_below(self):
        """
        Leaf nodes inherit bounds from their parent
        nodes and do not propagate further.
        """
        # Bounds are inherited from the parent node and remain unchanged
        pass

    def get_leaves_below(self):
        """
        Returns the leaf node itself in a list.
        """
        return [self]

    def pred(self, x):
        """
        Predict the class for a single individual at the leaf node.

        Parameters:
        x : np.ndarray
            A 1D NumPy array representing the features of a single individual.

        Returns:
        int
            The predicted class for the individual.
        """
        return self.value


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

    def count_nodes(self, only_leaves=False):
        """
        Counts the number of nodes in the decision tree.

        Parameters:
        only_leaves : bool, optional
            If True, count only the leaf nodes (default is False).

        Returns:
        int
            The number of nodes in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Returns a string representation of the decision tree.

        Returns:
        str
            The string representation of the decision tree.
        """
        return self.root.__str__() + "\n"

    def get_leaves(self):
        """
        Returns the list of all leaf nodes in the decision tree.

        Returns:
        list
            A list of all leaves in the tree.
        """
        return self.root.get_leaves_below()

    def get_leaves(self):
        """
        Returns a list of all leaf nodes in the decision tree.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Updates the bounds for all nodes in the decision tree.
        """
        self.root.update_bounds_below()

    def update_predict(self):
        """
        Updates the predict function for efficient batch predictions.
        """
        # Update bounds for each node
        self.update_bounds()

        # Get all the leaves
        leaves = self.get_leaves()

        # Update indicator for each leaf and store its contribution
        for leaf in leaves:
            leaf.update_indicator()

        # Define the efficient predict function
        self.predict = lambda A: np.sum(
            [leaf.indicator(A) * leaf.value for leaf in leaves], axis=0
        )

    def pred(self, x):
        """
        Predict the class for a single individual using the decision tree.

        Parameters:
        x : np.ndarray
            A 1D NumPy array representing the features of a single individual.

        Returns:
        int
            The predicted class for the individual.
        """
        return self.root.pred(x)
