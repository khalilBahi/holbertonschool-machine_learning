#!/usr/bin/env python3
""" Task 10: 10. IRF 1 : Isolation Random Trees """

import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree:
    """
    An Isolation Random Tree (IRT) is used for anomaly or outlier detection.
    The tree isolates outliers by randomly partitioning
    the data until points are isolated.
    The depth at which a point is isolated can
    be used to determine its outlier score.
    """

    def __init__(self, max_depth=10, seed=0, root=None):
        """
        Initializes the Isolation Random Tree with the specified parameters.

        Parameters:
        max_depth : int, optional
            Maximum depth of the tree (default is 10).
        seed : int, optional
            Seed for random number generation (default is 0).
        root : Node or Leaf, optional
            Root node of the tree (default is None, which creates a new Node).
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """
        Returns a string representation of the decision tree.

        Returns:
        str
            The string representation of the decision tree.
        """
        return self.root.__str__()

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
            If True, counts only the leaf nodes (default is False).

        Returns:
        int
            The number of nodes in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """
        Updates the bounds for the entire tree, starting from the root node.
        This is used to update the splitting criteria across the tree.

        Returns:
        None
        """
        return self.root.update_bounds_below()

    def get_leaves(self):
        """
        Returns a list of all the leaves in the tree.

        Returns:
        list
            A list of all the leaf nodes in the tree.
        """
        return self.root.get_leaves_below()

    def update_predict(self):
        """
        Updates the prediction function for the tree. After the tree is fit,
        this function calculates the sum of indicators across all leaf nodes.

        Returns:
        None
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            np.array([leaf.indicator(A) * leaf.value for leaf in leaves]),
            axis=0
        )

    def np_extrema(self, arr):
        """
        Returns the minimum and maximum values of an array.

        Parameters:
        arr : array-like
            The array from which the minimum and maximum values are derived.

        Returns:
        tuple
            The minimum and maximum values of the array.
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Determines a random split criterion for
        a given node by selecting a feature
        and computing the split threshold.

        Parameters:
        node : Node
            The node for which the split criterion is determined.

        Returns:
        tuple
            A tuple containing the feature index and the threshold value.
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(self.explanatory
                                                       [:, feature]
                                                       [node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """
        Creates and returns a leaf child node for the given parent node
        with the specified sub-population.

        Parameters:
        node : Node
            The parent node.
        sub_population : array-like
            The subpopulation of the data points assigned to this child node.

        Returns:
        Leaf
            A new leaf node with the updated sub-population and depth.
        """
        leaf_child = Leaf(node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Creates and returns a non-leaf child node for the given parent node
        with the specified sub-population.

        Parameters:
        node : Node
            The parent node.
        sub_population : array-like
            The subpopulation of the data points assigned to this child node.

        Returns:
        Node
            A new non-leaf node with the updated sub-population and depth.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """
        Recursively fits a node and its children based
        on random splits of the data.
        The function splits the data at the node
        and decides whether the split leads to
        a leaf node or another non-leaf node.

        Parameters:
        node : Node
            The node to fit.
        """
        node.feature, node.threshold = self.random_split_criterion(node)

        left_population = node.sub_population &\
            (self.explanatory[:, node.feature] > node.threshold)
        right_population = node.sub_population &\
            ~(self.explanatory[:, node.feature] > node.threshold)

        is_left_leaf = (node.depth == self.max_depth - 1)\
            or (np.sum(left_population) <= self.min_pop)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (node.depth == self.max_depth - 1)\
            or (np.sum(right_population) <= self.min_pop)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """
        Fits the entire Isolation Random Tree on the given explanatory data.

        Parameters:
        explanatory : array-like
            The explanatory variables used for training the tree.
        verbose : int, optional
            If set to 1, prints training statistics (default is 0).

        Returns:
        None
        """
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
