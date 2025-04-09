import pandas as pd
import numpy as np
import pytest

from variability_tree_splitter import (
    IOIsplitter,
    calculate_within_person_variability,
    calculate_between_person_variability,
)


@pytest.fixture
def toy_data():
    np.random.seed(0)
    subjects = np.repeat(np.arange(5), 4)  # 5 subjects, 4 visits each
    visits = list(range(4)) * 5
    df = pd.DataFrame(
        {
            "subject_id": subjects,
            "visit_number": visits,
            "feature1": np.random.normal(0, 1, 20),
            "feature2": np.random.normal(0, 1, 20),
            "age": np.repeat([30, 40, 35, 50, 45], 4),
            "sex": np.repeat([0, 1, 0, 1, 0], 4),
        }
    )
    return df


def test_within_person_variability(toy_data):
    wpv = calculate_within_person_variability(toy_data, ["feature1", "feature2"])
    assert isinstance(wpv, pd.Series)
    assert not wpv.isnull().any()


def test_between_person_variability(toy_data):
    bpv = calculate_between_person_variability(toy_data, ["feature1", "feature2"])
    assert isinstance(bpv, pd.Series)
    assert not bpv.isnull().any()


def test_ioi_splitter_fit(toy_data):
    splitter = IOIsplitter(
        data=toy_data,
        features=["feature1", "feature2"],
        covariates=["age", "sex"],
        min_subjects_per_leaf=2,
    )
    tree = splitter.fit()
    assert isinstance(tree, dict)
    assert "param" in tree
    assert "left" in tree
    assert "right" in tree


def test_assign_to_leaf(toy_data):
    splitter = IOIsplitter(
        data=toy_data,
        features=["feature1", "feature2"],
        covariates=["age", "sex"],
        min_subjects_per_leaf=2,
    )
    tree = splitter.fit()
    summary, _ = splitter.summarize_splits(tree)
    summary_df = pd.DataFrame(summary)

    # Assign all rows to leaves
    group_labels = toy_data.apply(
        lambda row: IOIsplitter.assign_to_leaf(summary_df, row), axis=1
    )
    assert isinstance(group_labels.iloc[0], str)
    assert group_labels.nunique() <= len(summary_df["node_id"].unique()) * 2
