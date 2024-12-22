#!/usr/bin/env python3
"""Task 9: 9. Random forests"""
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest:
    """
    A Random Forest classifier that aggregates predictions
    from multiple decision trees.

    Parameters:
    ----------
    n_trees : int, optional
        The number of trees in the forest. Default is 100.
    max_depth : int, optional
        The maximum depth of each tree. Default is 10.
    min_pop : int, optional
        The minimum population (number of samples) required
        to split a node. Default is 1.
    seed : int, optional
        The random seed for reproducibility. Default is 0.

    Attributes:
    ----------
    numpy_preds : list
        A list of trained decision trees in the forest.
    target : array-like
        The target values used for training.
    explanatory : array-like
        The input features used for training.
    """

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
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
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """
        Predicts the class for each example in 'explanatory'
        by aggregating the predictions from all trees.
        """
        # Initialize an empty list to store predictions from individual trees
        predictions = []

        # Generate predictions for each tree in the forest
        for tree in self.numpy_preds:
            predictions.append(tree.predict(explanatory))

        # Convert list of predictions into a numpy array for easier
        # manipulation
        predictions = np.array(predictions)

        # Calculate the mode (most frequent) prediction for each example
        # Axis 0 corresponds to trees, and Axis 1 corresponds to samples
        return np.array([np.bincount(predictions[:, i]).argmax()
                        for i in range(predictions.shape[1])])

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """
        Trains the Random Forest by creating 'n_trees' decision trees.
        """
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []

        depths = []
        nodes = []
        leaves = []
        accuracies = []

        for i in range(n_trees):
            T = Decision_Tree(
                max_depth=self.max_depth,
                min_pop=self.min_pop,
                seed=self.seed + i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T)  # Store the trained decision tree
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))

        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}
    - Accuracy of the forest on td   : {self.accuracy(self.explanatory,
    self.target)}""")

    def accuracy(self, test_explanatory, test_target):
        """
        Calculates the accuracy of the Random Forest on a test set.
        """
        return np.sum(np.equal(self.predict(test_explanatory),
                      test_target)) / test_target.size
