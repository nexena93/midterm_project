# Midterm project: classification of breast cancer samples 

## Problem description and how a model could be used
Breast cancer is one of the most common cancers worldwide, and early, accurate diagnosis is critical for patient outcomes. When a suspicious mass is found, a sample of cells is collected through a fine needle aspiration and examined under the microscope to determine whether the tumor is benign or malignant. This assessment is partly subjective and time-consuming, and it depends on the experience of the pathologist. 

* The problem addressed in this project is a *binary classification* task: given a set of quantitative measurements describing the cell nuclei in a digitized image of the sample (such as radius, perimeter, area, texture, smoothness, concavity and symmetry, each summarized as a mean, a standard error and a “worst” value), predict whether the tumor is malignant or benign. 

* A classification model trained on these features could serve as a decision-support tool for clinicians and pathologists. Once the nuclear measurements have been extracted from a new sample, the model would return the estimated probability that the tumor is malignant. This probability could be used to flag high-risk cases for priority review, to provide a second opinion that reduces inter-observer variability, and to help standardize the diagnostic process. 
* The model is intended to support, not replace, the clinician’s judgment: its output should be interpreted together with other clinical information and confirmed by standard diagnostic procedures, and it should be validated on independent external data before any real-world use.

## Dataset
The Wisconsin Diagnostic Breast Cancer dataset contains 569 analyzed samples (rows), each associated with a patient, and 32 columns: 
* 1 column with integer numbers `id`: it represents the numerical identifier of the breast mass sample.
* 1 column with string values `diagnosis`: it represents the diagnosis of the sample (`M`= malignant, `B`= benign).
* 30 columns with floating-point numbers describing the characteristics of cell nuclei observed in the sample images.

This dataset can be dowloaded from https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic.

## Data and modeling
Everything described below can be found in `notebook.ipynb`

### 1. Data preparation
Here I uploaded the dataset, read it with pandas and prepared it for machine learning:
-  Column names follow the structure described in `wdbc.names`: all of them are lowercase and spaces between words are replaced with underscores. 
- String values in `diagnosis` are M (malignant) or B (benign), so I lowercase them; for classification numbers are needed, so I compared the column with `m`, producing a boolean series and casted it to integers. Now `diagnosis` is 1 for malignant samples and 0 for those who are benign.

### 2. Validation framework
Before training anything, I split the data into train, validation and test set using Scikit-Learn and put the target variable `diagnosis` aside so it cannot leak into the features.
- I applied the classic 60/20/20 split: 60% of the data for training, 20% for validation and 20% for testing. The full dataset contains 569 samples: 341 for training, 114 for validation and 114 for testing.
- Scikit-Learn function `train_test_split` shuffles the rows, so after splitting the row indexes are a random mixture of numbers, so I reset them.
- The target of the project is `diagnosis` column, so I extracted it from each dataframe as a Numpy array and deleted it from the dataframes.

### 3. Exploratory Data Analysis (EDA)
Next I explored the training data, checking for missing values and looking at the distribution of the target variable. For EDA I used the full train set (train and validation together) in order to compute feature importance later.
- First thing, for every column, I counted the NaN values: in this dataset there aren't missing values.
- Secondly, I looked at the target variable: it is moderately imbalanced, resulting in roughly 63% of the benign samples and 37% of the malignant samples.
- Finally, I created a list of numerical columns, excluding `id` from the features, because it does not contain any biologically useful information for classification.

### 4. Feature importance
Here I identified which features affects most the target variable `diagnosis`, using the numerical variables.
- *Correlation*: it measures the strength and direction of a linear or monotonic relationship, indicating whether the relationship is positive or negative. The features most strongly associated with the diagnosis are `concave_points_worst` (0.81), `perimeter_worst` (0.80), and `radius_worst` (0.79), followed by the corresponding mean features; all the values listed fall between roughly 0.71 and 0.81, which indicates a strong association with the target. Features describing the size of the nucleus (radius, perimeter, area) and its shape irregularity (concavity, concave points) therefore appear to be the most informative for distinguishing between the two classes.
- *Mutual information*: tells how much I can learn about one variable if I know the value of another, capturing any kind of statistical dependence between two variables, linear or not. The features carrying the most information about the diagnosis are `perimeter_worst` (0.49), `area_worst` (0.48), and `radius_worst` (0.47), followed by `concave_points_worst` and `concave_points_mean`. The ranking is consistent with the one obtained from the correlation analysis, which suggests that these features, describing the size and the shape irregularity of the cell nuclei are the most relevant for distinguishing between the two classes.

### 5. Logistic regression
As a first model, I trained a logistic regression classifier (`LogisticRegression` from Scikit-Learn, with the `lbfgs` solver).
- Logistic regression models the probability that an observation belongs to class 1 as a function of a linear combination of the features, passed through a sigmoid function. 
- Since the features have very different scales (e.g. `area_worst` is in the thousands, whereas `smoothness` is below 1), I standardized them with a `StandardScaler`, rescaling each feature to zero mean and unit variance. This allows the `lbfgs` optimizer to converge, makes the regularization penalty act uniformly across features, and makes the model coefficients comparable. The scaler was fitted on the training data only and then applied to the validation and test sets to avoid data leakage.
- I created the model and called `fit`with the feature matrix `X_train_scaled` and the target vector `y_train`; for each observation in the validation set, the probabilities that the observation belongs to each class, as a matrix with one row per observation and one column per class. Since this is a binary classification problem, the matrix has two columns: the first contains the probability of class 0 and the second the probability of class 1. `y_pred` is a one-dimensional array containing the predicted probability that each validation observation belongs to the positive class (`diagnosis = 1`).
- If the model thinks the probability that the tumor is malignant is 50% or more, I predict that the tumor is malignant.
- To see whether these predictions are any good, I compared them with the actual labels of the validation set (`accuracy_score`): in about 98% of cases, the model predicted malignant or benign correctly.
- Finally I trained the model using the full feature matrix `X_full_train_scaled`and full target vector `y_full_train` and computed the predictions using the test set: accuracy on the test set is about 97%, so the model performed on unseen data about as well as on the data it was tuned with.

### 6. Evaluation metrics for classification
- I evaluated the accuracy on the validation set for 21 decision thresholds between 0 and 1; the two extreme thresholds correspond to trivial classifiers (always class 1 and always class 0) and yield accuracies equal to the class proportions (35.1% and 65.8%); accuracy reached its maximum (99.1%) for thresholds between 0.25 and 0.45, while the default threshold of 0.5 gave 98.2%. 
Because missing a malignant tumor is the most costly error in a medical setting, I decided to adopt the lower threshold of 0.25.
- Accuracy collapses all predictions into a single number, so it cannot tell us which kinds of mistakes the model makes. `confusion matrix` summarizes the model’s predictions by comparing them with the true labels; its rows represent the actual classes and its columns the predicted ones, so that the diagonal contains the correctly classified observations (true negatives and true positives), while the off-diagonal cells contain the errors: false positives (benign cases classified as malignant) and false negatives (malignant cases classified as benign).
On the validation set, the confusion matrix shows 74 true negatives, 0 false positives, 1 false negative and 39 true positives: the model correctly identified all benign cases (`specificity` = 100%) and every case it flagged as malignant was indeed malignant (`precision` = 100%) and it missed 2 of the 40 malignant cases (`recall` = 97.5%).
- Since false negatives are more costly than false positives in a medical setting, I used the `fbeta_score`, that is a weighted harmonic mean of precision and recall that gives recall twice the importance of precision.
 On the validation set, F2-score is 0.98, because one false negative was eliminated without introducing any false positive.
- To evaluate the discriminative ability of the model independently of the decision threshold, I computed the area under the ROC curve (`roc_auc_score`) using the predicted probabilities of class 1. The ROC curve plots the true positive rate against the false positive rate for all possible thresholds. On the validation set, the model reached a ROC AUC of 0.999, indicating an almost perfect ability to discriminate between malignant and benign tumors. This is consistent with the stable accuracy observed across a wide range of decision thresholds and with the very small number of misclassified observations in the confusion matrix. Because the ROC AUC is independent of the decision threshold, it complements the threshold-dependent metrics (recall, precision, F2-score), which were used to select the final operating point of 0.25.
- Hyperparameter tuning was performed with a grid search over the regularization strength `C` of the logistic regression (values 0.01, 0.1, 1, 10 and 100), using stratified 5-fold cross-validation (`StratifiedKFold`, in order to obtain a more robust estimate of model performance) on the full training set . Standardization was included in a pipeline, so that in each fold the scaler and the model were fitted on the training folds only. For each value of C I computed the mean and standard deviation of ROC AUC, together with recall and F2-score, and selected the best configuration according to the mean ROC AUC: `C=1` with a mean ROC AUC of 0.995. 

### 7. Random forest
As a second model, I trained a random forest classifier (`RandomForestClassifier`) using Scikit-Learn. 
- First I evaluated how its performance on the validation set changed with the number of trees (`n_estimators`), from 10 to 200 in steps of 10, using the ROC AUC (`roc_auc_score`)as the metric. The validation ROC AUC was already 0.999 with only 10 trees and remained essentially constant (between 0.9986 and 0.9993) as the number of trees increased, reaching a plateau from about 90 trees onward. Differences at the fourth decimal place are well within the noise expected from a validation set of 114 observations, so I fixed `n_estimators` = 100. The resulting AUC (0.9993) is practically identical to that of the logistic regression (0.999).
- Secondly, I evaluated the random forest for maximum depths (`max_depth`) of 5, 10 and 15 and for numbers of trees between 10 and 200. The validation ROC AUC was between 0.998 and 1.000 for all configurations, with differences confined to the third and fourth decimal places. Depths 10 and 15 gave virtually identical results, because the trees rarely reach a depth of 10 on a dataset of this size, and reached a stable plateau of 0.9993 from about 90 trees onward. The isolated maximum (AUC = 1.000 with max_depth = 5 and 20 trees) was not stable across neighboring values and was attributed to random variation rather than to a real improvement. I therefore selected `max_depth = 10` and `n_estimators = 100`.
- Then I evaluated the effect of the minimum number of samples per leaf (`min_samples_leaf`) of 1, 3, 5, 10, 50, with `max_depth` fixed at 10. The validation ROC AUC decreased monotonically as `min_samples_leaf` increased, from a stable plateau of 0.9993 for `min_samples_leaf` = 1 to about 0.9953 `for min_samples_leaf` = 50. Although all values yielded excellent discrimination (AUC > 0.995), larger leaves produce coarser probability estimates, which slightly worsens the ranking of observations. Since the dataset is highly separable and does not appear to suffer from overfitting, additional regularization provided no benefit, and we retained the default `min_samples_leaf = 1`. 
- The final random forest configuration was therefore `n_estimators = 100`, `max_depth = 10` and `min_samples_leaf = 1`.

### 8. XGBoost
As third and final model, I trained a XGBoost classifier (`XGBClassifier`) using Scikit-Learn.
- I explored learning rates (`learning_rate`) of 0.01, 0.05, 0.1 and 0.3 and maximum depths (`max_depth`) of 2, 3, 4 and 6. 
- Averaged over the other parameters, the validation ROC AUC increased from 0.997 (`learning_rate` = 0.01) to 0.999 (`learning_rate` = 0.3), with most of the gain obtained between 0.01 and 0.05, and from 0.998 (`max_depth` = 2) to 0.999 (`max_depth`= 4), with no further improvement at depth 6. The lowest learning rate was penalized because the number of trees explored was too small for such a slow update. 
- Since all differences beyond these thresholds were within the noise expected from a small validation set, we selected a simple configuration within the plateau: `learning_rate = 0.05`, `max_depth = 4`, `n_estimators = 100`. 

### 9. Selecting the best model
The three classifiers were compared using stratified 5-fold cross-validation on the full training set: logistic regression (mean ROC AUC = 0.995 ± 0.007), Random Forest (mean ROC AUC = 0.993 ± 0.005) and XGBoost (mean ROC AUC = 0.992 ± 0.005). 
- The differences between models were smaller than the fold-to-fold variability, so their discriminative performance can be considered equivalent. 
- Following the principle of parsimony, and since it also obtained the highest mean AUC, the logistic regression was selected as the final model, being simpler, faster and more interpretable than the tree-based alternatives. 
- The final model was refitted on the full training set (training + validation) and evaluated once on the independent test set. It achieved a ROC AUC of 0.996, an accuracy of 97.4% and an F2-score of 0.957 at a decision threshold of 0.25. The confusion matrix shows 71 true negatives, 1 false positive, 2 false negatives and 40 true positives, corresponding to a recall of 95.2%, a precision of 97.6% and a specificity of 98.6% for the malignant class. The test AUC is consistent with the cross-validated estimate (0.995), which suggests that the model generalizes well and that the selection procedure did not overfit the validation data.

### 10. Limitations
1. *Small sample size*: the dataset contains only a few hundred observations, and the independent test set includes 114 cases (42 malignant). As a consequence, each misclassified observation changes the reported metrics by roughly 1-2 percentage points, and the estimates carry a non-negligible degree of uncertainty. Differences between models (for example, in cross-validated ROC AUC) were smaller than the fold-to-fold variability and should not be interpreted as evidence of real superiority.
2. *Single train/test split*: the final performance was estimated on one held-out test set. A different random split could have produced somewhat different results. Confidence intervals (for instance, obtained by bootstrapping the test set) were not computed, so the precision of the reported estimates is not quantified.
3. *Residual false negatives*: even with a lowered threshold, the final model missed 2 of the 42 malignant cases in the test set. In a clinical setting, missing a malignant tumor is the most costly type of error.
4. *No external validation*: all data come from a single dataset, so the model’s ability to generalize to other populations, imaging devices or laboratories was not assessed. For this reason, the model should not be used for clinical decision-making without validation on independent external data.

## Project files
Besides `notebook.ipynb`, the project contains the following files, which turn the final model into a deployed web service:

### `train.py`
This script trains the final model and saves it to disk. It reproduces the final steps of the notebook:
1. Loads the dataset (`/workspaces/midterm_project/wdbc.data`), assigns the column names and encodes the target (`M` = 1 malignant, `B` = 0 benign).
2. Splits the data into a full training set (80%) and a test set (20%), using the same `random_state` as the notebook.
3. Trains the selected model: a pipeline made of a `StandardScaler` and a `LogisticRegression`.
4. Evaluates it on the test set (ROC AUC and F2-score).
5. Saves the trained pipeline, the list of features and the decision threshold (0.25) to `model.bin` with `pickle`.

```bash
uv run python train.py
```

### `predict.py`
This is the web service built with **FastAPI** that loads `model.bin` and exposes the model through an HTTP API:
- `POST /predict`: receives a JSON object with the 30 numerical features of a tumor and returns the probability of malignancy, the predicted class (`diagnosis`: 1 or 0), its label (`malignant` / `benign`) and the threshold used.
- `GET /health`: health check, returns `{"status": "ok"}`.
- `GET /docs`: interactive documentation (Swagger UI) where the API can be tested from the browser.

The input is validated with Pydantic: if a feature is missing or is not numeric, the service returns a `422` error describing the problem.

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

### `predict_test.py`
This is a simple client script to test the service; it reads one malignant and one benign example from the dataset, sends them to the `/predict` endpoint and prints the responses. The URL of the service can be passed as an argument (default: `http://localhost:9696`).

```bash
uv run python predict_test.py                                  
uv run python predict_test.py https://breast-cancer-predict.fly.dev  
```

### `pyproject.toml` and `uv.lock`
Dependency management with **uv**:
- `pyproject.toml` lists the direct dependencies. The libraries needed by the service (scikit-learn, pandas, numpy, FastAPI, uvicorn) are separated from the development ones (Jupyter kernel, matplotlib, XGBoost, requests), which are only needed for the notebook and the tests.
- `uv.lock` pins the exact version of every package, so that the environment is reproducible on any machine and inside Docker. This is important because a pickled scikit-learn model should be loaded with the same library versions used to train it.

```bash
uv sync
```

### `Dockerfile`
Defines the image that runs the service:
1. Starts from `python:3.13-slim` and adds `uv`.
2. Installs the locked production dependencies with `uv sync --locked --no-dev` (copied before the code, so Docker can cache this layer).
3. Copies only `predict.py` and `model.bin`.
4. Starts the service with `uvicorn` on port 9696, or on the port given by the `PORT` environment variable when set by the hosting platform.

```bash
docker build -t breast-cancer-service .
docker run -it --rm -p 9696:9696 breast-cancer-service
```
A `.dockerignore` file excludes the virtual environment, the data and the notebook from the image.

### Deployment
The Docker image is deployed to the cloud on **Fly.io**:

```bash
fly launch   #detects the Dockerfile and creates fly.toml
fly deploy
```

The service is available at: **https://breast-cancer-predict.fly.dev**

- Interactive documentation: https://breast-cancer-predict.fly.dev/docs
- Health check: https://breast-cancer-predict.fly.dev/health

Example request:

```bash
curl -X POST https://breast-cancer-predict.fly.dev/predict \
  -H "Content-Type: application/json" \
  --data-binary @examples/malignant.json
```