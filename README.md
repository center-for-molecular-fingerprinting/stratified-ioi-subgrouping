# stratified-ioi-subgrouping

A custom decision-tree-like splitter that partitions a dataset based on maximizing the **Index of Individuality (IOI)**, defined as the ratio of within-person to between-person variability.

This method is useful for exploring how covariates influence stability and variability in longitudinal data and for stratifying subjects into more homogeneous subgroups.

---

## 🚀 Features

- 📈 Maximizes IOI to find meaningful subgroup splits.
- 🔀 Supports both continuous and binary covariates.
- 🧑‍🤝‍🧑 Enforces minimum subject count per group to avoid overfitting.
- 🌲 Builds a recursive tree structure for splits.
- 🧩 Assigns new samples to subgroups using the fitted tree.
- 📊 Provides tools for summarizing the tree and computing IOI within groups.

---

## 🧠 What is IOI?

> IOI = Within-Person Variability / Between-Person Variability

A higher IOI indicates that within-person variability is greater than between-person variability.  
Lowering IOI through covariate-based grouping suggests more stable subject behavior within subgroups.

---

## ⚙️ Installation

You can copy the `variability_tree_splitter.py` file into your project or install as a local module.  
Dependencies:
- `pandas`
- `numpy`
- (optionally `pytest` for testing)

---

## 🧪 Example Usage

A full walkthrough is available in [`example_usage.ipynb`](example_usage.ipynb).  
It shows how to:
- Preprocess the data to ensure subject-level consistency
- Train the splitter
- Summarize the resulting tree
- Assign each sample to a group
- Compute IOI within groups

Here’s a minimal snippet:

```python
from variability_tree_splitter import IOIsplitter
import pandas as pd

data = pd.read_csv("your_data.csv")
features = [str(i) for i in range(519)]  # e.g. wavenumber features
covariates = ["mean_age", "mean_bmi", "sex"]

splitter = IOIsplitter(data, features, covariates, min_subjects_per_leaf=10)
tree = splitter.fit()
summary, _ = splitter.summarize_splits(tree)

# Assign each row to a leaf
summary_df = pd.DataFrame(summary)
data["group"] = data.apply(lambda row: IOIsplitter.assign_to_leaf(summary_df, row), axis=1)
```

## 🤝 Contributing

Contributions are welcome and appreciated! ✨

If you'd like to help improve **stratified-ioi-subgrouping**, here's how to get started:

### 🛠️ How to Contribute

1. **Fork** this repository
2. **Clone** your fork:
   ```bash
   git clone https://github.com/your-username/stratified-ioi-subgrouping.git
   cd stratified-ioi-subgrouping
   ```
3. **Create a new branch** for your changes:
   ```bash
   git checkout -b feature/my-awesome-change
   ```
4. Make your changes and be sure to:
    - Follow existing code style  
    - Include docstrings and type hints where relevant  
    - Add or update tests if needed
5. **Commit** your changes:
    ```bash
    git commit -m "Add: brief description of your change"
    ```
6. **Push** to your fork:
    ```bash
    git push origin feature/my-awesome-change
    ```

7. **Open a Pull Request** from your branch to main in this repo 🎉

### 📋 Guidelines

- Use clear, descriptive commit messages.
- Keep pull requests focused and minimal — small, isolated changes are easier to review.
- Follow existing code formatting and structure.
- Include docstrings and type hints where applicable.
- Add or update unit tests to maintain test coverage.
- If you're not sure about a change, open an issue or discussion before submitting a PR.

---

### 🧪 Testing

Before submitting a pull request, please run the test suite locally:

```bash
pytest test_ioi_splitter.py
``` 

## 📝 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
