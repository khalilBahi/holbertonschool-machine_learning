#!/usr/bin/env python3
"""Task 9: 9. Random forests"""
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest:
    """
    A Random Forest classifier that aggregates
    predictions from multiple decision trees.

    Parameters:
    ----------
    n_trees : int, optional
        The number of trees in the forest. Default is 100.
    max_depth : int, optional
        The maximum depth of each tree. Default is 10.
    min_pop : int, optional
        The minimum population (number of samples)
        required to split a node. Default is 1.
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
        Initializes the Random Forest with the specified parameters.

        Parameters:
        ----------
        n_trees : int, optional
            The number of trees in the forest. Default is 100.
        max_depth : int, optional
            The maximum depth of each tree. Default is 10.
        min_pop : int, optional
            The minimum population (number of samples)
            required to split a node. Default is 1.
        seed : int, optional
            The random seed for reproducibility. Default is 0.
        """
        self.numpy_preds = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """
        Predicts the class for each example in the given explanatory data.

        Aggregates the predictions from all trees in the
        forest by taking the mode (most frequent)
        prediction across the trees for each example.

        Parameters:
        ----------
        explanatory : array-like
            The input data for which to make predictions.

        Returns:
        -------
        np.ndarray
            Array of predicted classes for each input example.
        """
        predictions = []
        for tree in self.numpy_preds:
            predictions.append(tree.predict(explanatory))

        predictions = np.array(predictions)
        return np.array([np.bincount(predictions[:, i]).argmax()
                        for i in range(predictions.shape[1])])

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """
        Trains the Random Forest by creating 'n_trees' decision trees.

        Parameters:
        ----------
        explanatory : array-like
            The input features used for training the model.
        target : array-like
            The target values used for training the model.
        n_trees : int, optional
            The number of trees to train. Default is 100.
        verbose : int, optional
            If 1, prints training progress. Default is 0.
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
            self.numpy_preds.append(T)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))

        if verbose == 1:
            print(f"""Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}
    - Accuracy of the forest on td   : {self.accuracy(self.explanatory,
    self.target)}""")

    def accuracy(self, test_explanatory, test_target):
        """
        Calculates the accuracy of the Random Forest on a test set.

        Parameters:
        ----------
        test_explanatory : array-like
            The test input data.
        test_target : array-like
            The true labels for the test data.

        Returns:
        -------
        float
            The accuracy of the model on the test data.
        """
        return np.sum(np.equal(self.predict(test_explanatory),
                      test_target)) / test_target.size
