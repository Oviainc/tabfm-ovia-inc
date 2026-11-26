# Ovia Inc — TabFM (fork of google-research/tabfm)

We maintain this fork of [google-research/tabfm](https://github.com/google-research/tabfm), TabFM (Tabular Foundation Model): a pretrained, scikit-learn compatible foundation model for tabular data that performs zero-shot classification and regression on datasets with mixed column types, with no training on the target dataset at inference time. Upstream credit: Google Research and the TabFM contributors. Licence: Apache-2.0 (see LICENSE).

**Our main site:** https://oviainc.github.io/ai-services/

**Upstream repository:** https://github.com/google-research/tabfm

**Upstream introduction:** https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/

## What this is

The tabular-model seat of the infrastructure and data group in our collection:

- The complete upstream project at the fork point (Python, the `tabfm` package, `examples/`, `results/`, and `requirements.txt`): a pretrained model that reads training rows as context and predicts on new samples through in-context learning, rather than fitting parameters per dataset.
- Two interchangeable backends preserved in the code: JAX (Flax `nnx`) and PyTorch, with the pretrained TabFM v1.0.0 weights downloaded and loaded automatically through Hugging Face Hub.
- A scikit-learn compatible surface — `TabFMClassifier` and `TabFMRegressor` with `fit`, `predict`, and `predict_proba` — handling mixed numerical and categorical features, ordinal encoding, and numerical scaling inside the familiar estimator shape.
- Runnable examples for classification and regression, including TabArena variants, and recorded evaluation results in `results/` as parquet tables from the fork point.
- No picture is added in this pass: the repository holds no image files of its own, so no header image exists to use from the project's own assets. A social preview candidate is noted in the prepared repository details only.

Upstream carries about 2,860 stars at the time of this pass. Upstream records the project as not an officially supported Google product, preserved in the upstream text below.

## Story

TabFM opened as a public project under google-research on June 16, 2026, and reached its initial 1.0.0 release on June 29, 2026, per the changelog preserved in this repository. The shape is distinct on the model shelf: most foundation models in the collection work on text, images, or video, while business data most often arrives as tables — rows, columns, mixed types, missing structure. TabFM treats that table as context. Training examples go in as context at inference time, and predictions come out without a per-dataset training run, in the same in-context pattern used by language models, applied to regression and classification.

We forked the project on July 2, 2026, at the July 1, 2026 snapshot (tip commit 05f2c8e, the merge fixing classifier prediction label types), into the infrastructure and other group of the Oviainc collection. In role, this fork is the tabular foundation model seat: the place where zero-shot prediction on structured business data gets evaluated beside the conventional per-dataset modelling route held elsewhere in the collection, including our own data-science and machine-learning-engineering repositories.

What we evaluate here is the zero-shot claim in practice, with its limits kept visible:

- **Zero-shot quality against a trained baseline** — recorded TabArena results in `results/` are the starting claim; the test that matters is accuracy and calibration on ordinary business tables against a gradient-boosted or scikit-learn baseline trained on the same rows.
- **Context limits** — in-context learning means the training rows travel with each prediction; how many rows and columns fit usefully, and what sampling or context caching (added upstream after the fork point) changes, decides the usable dataset size.
- **Backend shape** — JAX and PyTorch builds of the same model allow a direct comparison of install weight, speed, and hardware fit before any deployment choice.
- **Mixed-type handling** — categorical and numerical columns arrive together in real tables; the encoding and scaling behaviour inside the estimator decides how much preparation work remains outside the model.

Where the project fits our work:

- **Business intelligence and data** — the direct fit: tabular prediction is the core of forecasting, scoring, churn, pricing, and operational reporting work; a foundation model that predicts on a new table without a training project changes the cost of a first answer.
- **AI agents** — agent systems that answer questions over business data need a prediction step as well as a language step; a scikit-learn shaped tabular model is a tool an agent workflow can call with a table and get a scored result back.
- **Context engineering** — TabFM is in-context learning outside language: the training table is the context, and its size, ordering, and selection shape the prediction; the same discipline of what goes into context, and what stays out, applies directly.
- **Virtual production** — a lighter fit: production and scheduling data is tabular, and quick zero-shot estimates on budgets, timelines, and resource tables are the plausible evaluation surface, not the media pipeline itself.

## Modifications by Ovia Inc

Documentation only. Upstream code and content are unchanged from the fork point, except for documentation notes added in November 2026: a modification reference on November 4, 2026, replaced by this expanded treatment (Ovia section, story, and repository details) on November 26, 2026. Where a later commit in this repository says otherwise, that commit governs.

## Sync status

Behind upstream, by design of this pass. This fork carries the July 1, 2026 snapshot (tip commit 05f2c8e) plus documentation commits of our own (November 4, 2026, replaced and expanded November 26, 2026). A cross-repository compare at this pass records the fork as diverged, ahead by 1 documentation commit and behind by 66 commits, at merge base 05f2c8e, with upstream HEAD dc64ff2 pushed November 20, 2026, including documentation of context caching options after the fork point. No sync was applied in this pass, and a reviewed sync is recommended for Ja when the data shelf needs a current upstream base.

## Related repositories

- [ai-services](https://github.com/Oviainc/ai-services) — our flagship site: AI agents, machine learning in production, and AI products for business
- [data-science](https://github.com/Oviainc/data-science) — our own data-science foundation repository, held beside this fork for the conventional modelling comparison
- [machine-learning-engineering](https://github.com/Oviainc/machine-learning-engineering) — our own machine-learning engineering foundation repository: production practice beside foundation-model evaluation
- [mcp-toolbox-ovia-inc](https://github.com/Oviainc/mcp-toolbox-ovia-inc) — the data-access layer of the agent collection: database tools that feed tabular work
- [xy-ovia-inc](https://github.com/Oviainc/xy-ovia-inc) — charting for data work in the collection, the presentation side of the same BI and data shelf

## Working with us

Start from our site: https://oviainc.github.io/ai-services/ — the contact form there is the fastest way to reach us.

## Licence

Apache-2.0, as recorded in the LICENSE file in this fork and upstream. Upstream credit remains with Google Research and the TabFM contributors.

---

Ovia Inc — Artificial Intelligence & Context Engineering. Based in Canada, working wherever the work is.

---
# TabFM: Tabular Foundation Models

TabFM (Tabular Foundation Model) is a scikit-learn compatible tabular foundation model. It allows you to perform zero-shot classification and regression on tabular datasets with mixed column types out-of-the-box.

At inference time, TabFM does not require training parameters on your dataset; instead, it leverages in-context learning by reading your training data as "context" to make instant predictions on new test samples.

*This is not an officially supported Google product.*

---

## Installation

To install TabFM, clone the repository and install it locally with the backend of your choice:

**JAX (CPU):**
```bash
git clone https://github.com/google-research/tabfm.git
cd tabfm
pip install -e .[jax]
```

**JAX (GPU):**
```bash
git clone https://github.com/google-research/tabfm.git
cd tabfm
pip install -e .[jax,cuda]
```

**PyTorch (CPU/GPU):**
```bash
git clone https://github.com/google-research/tabfm.git
cd tabfm
pip install -e .[pytorch]
```
*Note: For PyTorch with GPU support, ensure you have the appropriate PyTorch version installed for your CUDA version before installing TabFM.*

### Requirements
For a complete list of pinned dependencies and versions, please see [requirements.txt](requirements.txt). The core requirements depend on the backend you choose:
*   Python >= 3.11
*   Hugging Face Hub (for downloading pre-trained weights)
*   **JAX Backend:**
    *   JAX (specifically `jax==0.10.1`)
    *   Flax (specifically `flax==0.12.7`, using the modern `flax.nnx` API)
*   **PyTorch Backend:**
    *   PyTorch (specifically `torch==2.12.1+cpu` or a GPU version)

---

## Quick Start (TabFM v1.0.0)

We provide pre-trained weights for the **TabFM v1.0.0** release. The library handles downloading and loading these weights automatically. You can choose to load the model using either the JAX or PyTorch backend.

### 1. Classification Example

```python
import numpy as np
import pandas as pd
from tabfm import TabFMClassifier

# Choose your backend:

# OPTION A: JAX Backend
from tabfm import tabfm_v1_0_0_jax as tabfm_v1_0_0
model = tabfm_v1_0_0.load()

# OPTION B: PyTorch Backend
# from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
# model = tabfm_v1_0_0.load()

# Initialize scikit-learn compatible classifier (works with either backend model)
clf = TabFMClassifier(model=model)

# Prepare your dataset (supports mixed numerical and categorical features)
X_train = pd.DataFrame({
    "age": [25.0, 45.0, 35.0, 50.0],
    "job": ["engineer", "manager", "engineer", "manager"],
    "income": [80000, 120000, 90000, 130000]
})
y_train = np.array(["low_risk", "high_risk", "low_risk", "high_risk"])

X_test = pd.DataFrame({
    "age": [30.0, 48.0],
    "job": ["engineer", "manager"],
    "income": [85000, 125000]
})

# Fit classifier (prepares ordinal encoders and numerical scalers)
clf.fit(X_train, y_train)

# Predict classes and probabilities
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

print("Predictions:", predictions)
print("Class Probabilities:\n", probabilities)
```

### 2. Regression Example

```python
import numpy as np
import pandas as pd
from tabfm import TabFMRegressor

# Choose your backend:

# OPTION A: JAX Backend
from tabfm import tabfm_v1_0_0_jax as tabfm_v1_0_0
model = tabfm_v1_0_0.load()

# OPTION B: PyTorch Backend
# from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
# model = tabfm_v1_0_0.load()

# Initialize scikit-learn compatible regressor (works with either backend model)
reg = TabFMRegressor(model=model)

# Prepare your dataset
X_train = pd.DataFrame({
    "sqft": [1200, 2500, 1500, 3000],
    "neighborhood": ["A", "B", "A", "C"]
})
y_train = np.array([250000, 550000, 310000, 620000])

X_test = pd.DataFrame({
    "sqft": [1800, 2800],
    "neighborhood": ["A", "B"]
})

# Fit and Predict
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

print("Predicted Prices:", predictions)
```

---

## Examples Directory

You can find runnable scripts for both classification and regression under the [examples/](examples/) folder:
*   [classification_example.py](examples/classification_example.py)
*   [regression_example.py](examples/regression_example.py)

To run them, simply execute:
```bash
python examples/classification_example.py
```
*(You can edit these files to switch between JAX and PyTorch backends as shown in the comments inside them).*

---

## Evaluation Results

Our model evaluation results can be found in [results/](results/).

---

## Running Tests

You can run the unit tests directly using Python's `unittest` module:

```bash
# Run all tests (requires both JAX and PyTorch installed)
PYTHONPATH=. python3 -m unittest discover -s tabfm/src/ -p "*_test.py"

# Or run specific test files:
PYTHONPATH=. python3 -m unittest tabfm/src/pytorch/model_test.py
PYTHONPATH=. python3 -m unittest tabfm/src/classifier_and_regressor_pytorch_test.py
```

Alternatively, if you have Bazel installed, you can run tests with:
```bash
bazel test //...
```
