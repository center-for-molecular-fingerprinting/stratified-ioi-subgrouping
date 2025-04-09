from typing import List, Union, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np


def calculate_within_person_variability(
    group: pd.DataFrame, features: List[Union[str, float]]
) -> pd.Series:
    """
    Calculates the within-person variability by computing the standard deviation
    of each feature per subject and taking the mean.

    Parameters:
        group (pd.DataFrame): The input data grouped by subject.
        features (list of str): List of feature column names to compute variability for.

    Returns:
        float: Mean within-person standard deviation across features.
    """
    return group.groupby("subject_id", observed=False)[features].std().mean()


def calculate_between_person_variability(
    group: pd.DataFrame, features: List[Union[str, float]]
) -> pd.Series:
    """
    Calculates the between-person variability by computing the standard deviation
    of each feature per visit and taking the mean.

    Parameters:
        group (pd.DataFrame): The input data grouped by visit number.
        features (list of str): List of feature column names to compute variability for.

    Returns:
        float: Mean between-person standard deviation across features.
    """
    return group.groupby("visit_number", observed=False)[features].std().mean()


def calculate_ioi(
    group: pd.DataFrame, features: List[Union[str, float]]
) -> pd.DataFrame:
    """
    Computes the Index of Individuality (IOI) for a given group by taking the ratio
    of within-person variability to between-person variability for each feature.

    Parameters:
        group (pd.DataFrame): The input data to compute IOI on.
        features (list): Feature columns to include in the calculation.

    Returns:
        pd.DataFrame: A one-row DataFrame containing IOI values for each feature.
    """
    bpv_group = calculate_between_person_variability(group, features)
    wpv_group = calculate_within_person_variability(group, features)
    ioi = pd.DataFrame(
        np.expand_dims((wpv_group / bpv_group).values, axis=0), columns=features
    )
    return ioi


class IOIsplitter:
    """
    A class to perform splitting of a dataset to maximize Index of Individuality (IOI),
    which is defined as within-person variability divided by between-person variability.

    Attributes:
        data (pd.DataFrame): Input dataset.
        features (list): Feature columns used to calculate IOI.
        covariates (list of str): Covariates to consider for splitting.
        min_subjects_per_leaf (int): Minimum unique subjects required in a leaf.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        features: List[Any],
        covariates: List[str],
        min_subjects_per_leaf: int = 10,
    ):
        """
        Initializes the IOIsplitter.

        Parameters:
            data (pd.DataFrame): The full dataset.
            features (list): Features for IOI computation.
            covariates (list of str): Covariates to split on.
            min_subjects_per_leaf (int): Minimum number of unique subjects per split.
        """
        self.data = data
        self.min_subjects_per_leaf = min_subjects_per_leaf
        self.covariates = covariates
        self.features = features

    def calculate_ioi(self, group: pd.DataFrame) -> float:
        """
        Computes the IOI for a given group of data.

        Parameters:
            group (pd.DataFrame): Subset of the data.

        Returns:
            float: Mean IOI across features.
        """
        wpv = calculate_within_person_variability(group, self.features)
        bpv = calculate_between_person_variability(group, self.features)
        ioi = wpv / bpv
        return ioi.mean()

    def find_best_split(
        self, group: pd.DataFrame
    ) -> Tuple[Optional[str], Optional[float], float]:
        """
        Identifies the best covariate and split value that maximizes the IOI.

        Parameters:
            group (pd.DataFrame): Subset of data to split.

        Returns:
            tuple: (best_param, best_split_value, best_ioi)
        """

        def _count_unique_subjects(group: pd.DataFrame) -> int:
            return group["subject_id"].nunique()

        def _weighted_objective(left: pd.DataFrame, right: pd.DataFrame) -> float:
            left_objective = self.calculate_ioi(left)
            right_objective = self.calculate_ioi(right)
            left_subjects = _count_unique_subjects(left)
            right_subjects = _count_unique_subjects(right)
            total_subjects = left_subjects + right_subjects
            return (
                left_subjects / total_subjects * left_objective
                + right_subjects / total_subjects * right_objective
            )

        best_ioi = float("-inf")
        best_param: Optional[str] = None
        best_split: Optional[float] = None

        for param in self.covariates:
            if param == "sex":
                left_split = group[group[param] == 0]
                right_split = group[group[param] == 1]
                if (
                    _count_unique_subjects(left_split) < self.min_subjects_per_leaf
                    or _count_unique_subjects(right_split) < self.min_subjects_per_leaf
                ):
                    continue
                split_ioi = _weighted_objective(left_split, right_split)

                if split_ioi > best_ioi:
                    best_ioi = split_ioi
                    best_param = param
                    best_split = 0.5
            else:
                unique_values = group[param].unique()
                for split_val in unique_values:
                    left_split = group[group[param] <= split_val]
                    right_split = group[group[param] > split_val]

                    if (
                        _count_unique_subjects(left_split) < self.min_subjects_per_leaf
                        or _count_unique_subjects(right_split)
                        < self.min_subjects_per_leaf
                    ):
                        continue

                    split_ioi = _weighted_objective(left_split, right_split)

                    if split_ioi > best_ioi:
                        best_ioi = split_ioi
                        best_param = param
                        best_split = split_val

        return best_param, best_split, best_ioi

    def split_data(
        self, group: pd.DataFrame, depth: int = 0
    ) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Recursively splits the data to form a tree structure that maximizes IOI.

        Parameters:
            group (pd.DataFrame): Subset of data to split.
            depth (int): Depth of the current node.

        Returns:
            dict or pd.DataFrame: Tree node or leaf data if splitting stops.
        """
        if group["subject_id"].nunique() < 2 * self.min_subjects_per_leaf:
            return group

        param, split, ioi = self.find_best_split(group)
        if param is None:
            print("param is None", depth)
            return group

        if param == "sex":
            left_split = group[group[param] == 0]
            right_split = group[group[param] == 1]
        else:
            left_split = group[group[param] <= split]
            right_split = group[group[param] > split]

        left_branch = self.split_data(left_split, depth + 1)
        right_branch = self.split_data(right_split, depth + 1)

        return {
            "param": param,
            "split": split,
            "ioi": ioi,
            "left": left_branch,
            "right": right_branch,
        }

    def fit(self) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Builds the full decision tree from the input data.

        Returns:
            dict: Tree structure with recursive splits.
        """
        return self.split_data(self.data)

    def summarize_splits(
        self,
        tree: Union[Dict[str, Any], pd.DataFrame],
        depth: int = 0,
        node_id: int = 1,
        parent_id: Optional[int] = None,
        direction: Optional[str] = None,
        summaries: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Flattens the tree structure into a list of node metadata for easier analysis.

        Parameters:
            tree (dict): Nested tree dictionary from `fit()`.
            depth (int): Current node depth.
            node_id (int): ID of the current node.
            parent_id (int): ID of the parent node.
            direction (str): Split direction ('left' or 'right').
            summaries (list): Accumulator for summaries.

        Returns:
            tuple: (summaries list, next available node_id)
        """
        if summaries is None:
            summaries = []

        if not isinstance(tree, dict) or "param" not in tree:
            return summaries, node_id

        summaries.append(
            {
                "node_id": node_id,
                "depth": depth,
                "param": tree["param"],
                "split_value": tree["split"],
                "ioi": tree["ioi"],
                "parent_id": parent_id,
                "direction": direction,
            }
        )

        next_node_id = node_id + 1

        summaries, next_node_id = self.summarize_splits(
            tree["left"], depth + 1, next_node_id, node_id, "left", summaries
        )
        summaries, next_node_id = self.summarize_splits(
            tree["right"], depth + 1, next_node_id, node_id, "right", summaries
        )

        return summaries, next_node_id

    @staticmethod
    def assign_to_leaf(summary_df: pd.DataFrame, data_row: pd.Series) -> str:
        """
        Assigns a data row to a leaf node by traversing the tree structure.

        Parameters:
            summary_df (pd.DataFrame): Flattened tree summary from `summarize_splits`.
            data_row (pd.Series): A single row of input data.

        Returns:
            str: Leaf node identifier (e.g., "3_left", "5_right").
        """
        current_node = summary_df[summary_df["depth"] == 0].iloc[0]
        while True:
            children = summary_df[summary_df["parent_id"] == current_node["node_id"]]
            param = current_node["param"]
            if param == "sex":
                direction = "left" if data_row[param] == 0 else "right"
            else:
                direction = (
                    "left"
                    if data_row[param] <= current_node["split_value"]
                    else "right"
                )

            next_node_df = children[children["direction"] == direction]

            if next_node_df.empty:
                return f'{current_node["node_id"]}_{direction}'

            current_node = next_node_df.iloc[0]
