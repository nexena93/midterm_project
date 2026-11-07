# Midterm project: classification of breast cancer samples 

## Problem

## Dataset
The Wisconsin Diagnostic Breast Cancer dataset contains 569 analyzed samples (rows), each associated with a patient, and 32 columns: 
* 1 column with integer numbers `id`: it represents the numerical identifier of the breast mass sample.
* 1 column with string values `diagnosis`: it represents the diagnosis of the sample (`M`= malignant, `B`= benign)
* 30 columns with floating-point numbers describing the characteristics of cell nuclei observed in the sample images:
    * `radius_mean`:
    * `texture_mean`:
    * `perimeter_mean`:
    * `area_mean`:
    * `smoothness_mean`:
    * `compactness_mean`:
    * `concavity_mean`:
    * `concave_points_mean`:
    * `symmetry_mean`:
    * `fractal_dimension_mean`:
    * `radius_se`:
    * `texture_se`:
    * `perimeter_se`:
    * `area_se`:
    * `smoothness_se`:
    * `compactness_se`:
    * `concavity_se`:
    * `concave_points_se`:
    * `symmetry_se`:
    * `fractal_dimension_se`:
    * `radius_worst`:
    * `texture_worst`:
    * `perimeter_worst`:
    * `area_worst`:
    * `smoothness_worst`:
    * `compactness_worst`:
    * `concavity_worst`:
    * `concave_points_worst`:
    * `symmetry_worst`:
    * `fractal_dimension_worst`:

This dataset can be dowloaded from https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic.

## Data preparation
Here I uploaded the dataset, read it with pandas and prepared it for machine learning:
-  Column names follow the structure described in `wdbc.names`: all of them are lowercase and spaces between words are replaced with underscores. 
- String values in `diagnosis` are M (malignant) or B (benign), so I lowercase them; for classification numbers are needed, so I compared the column with `m`, producing a boolean series and casted it to integers. Now `diagnosis` is 1 for malignant samples and 0 for those who are benign.

## Validation framework
Before training anything, I split the data into train, validation and test set using Scikit-Learn and put the target variable `diagnosis` aside so it cannot leak into the features.
- I applied the classic 60/20/20 split: 60% of the data for training, 20% for validation and 20% for testing. The full dataset contains 569 samples: 341 for training, 114 for validation and 114 for testing.
- Scikit-Learn function `train_test_split` shuffles the rows, so after splitting the row indexes are a random mixture of numbers, so I reset them.
- The target of the project is `diagnosis` column, so I extracted it from each dataframe as a Numpy array and deleted it from the dataframes.

## Exploratory Data Analysis (EDA)
Next I explored the training data, checking for missing values and looking at the distribution of the target variable. For EDA I used the full train set - train and validation together - in order to compute feature importance later.
- First thing, for every column, I counted the NaN values: in this dataset there aren't missing values.
- Secondly, I looked at the target variable: it is moderately imbalanced, resulting in roughly 63% of the benign samples and 37% of the malignant samples.
- Finally, I created a list of numerical columns, excluding `id` from the features, because it does not contain any biologically useful information for classification.

## Feature importance
Here I identified which features affects most the target variable `diagnosis`, using the numerical variables.
- Correlation: it measures the strength and direction of a linear or monotonic relationship, indicating whether the relationship is positive or negative. The features most strongly associated with the diagnosis are `concave_points_worst` (0.81), `perimeter_worst` (0.80), and `radius_worst` (0.79), followed by the corresponding mean features; all the values listed fall between roughly 0.71 and 0.81, which indicates a strong association with the target. Features describing the size of the nucleus (radius, perimeter, area) and its shape irregularity (concavity, concave points) therefore appear to be the most informative for distinguishing between the two classes.
- Mutual information: tells how much I can learn about one variable if I know the value of another, capturing any kind of statistical dependence between two variables, linear or not. The features carrying the most information about the diagnosis are `perimeter_worst` (0.49), `area_worst` (0.48), and `radius_worst` (0.47), followed by `concave_points_worst` and `concave_points_mean`. The ranking is consistent with the one obtained from the correlation analysis, which suggests that these features, describing the size and the shape irregularity of the cell nuclei are the most relevant for distinguishing between the two classes.

## Logistic regression
As a first model, I trained a logistic regression classifier (`LogisticRegression` from Scikit-Learn, with the `lbfgs` solver).
- Logistic regression models the probability that an observation belongs to class 1 as a function of a linear combination of the features, passed through a sigmoid function. 
- Since the features have very different scales (e.g. `area_worst` is in the thousands, whereas `smoothness` is below 1), we standardized them with a `StandardScaler`, rescaling each feature to zero mean and unit variance. This allows the `lbfgs` optimizer to converge, makes the regularization penalty act uniformly across features, and makes the model coefficients comparable. The scaler was fitted on the training data only and then applied to the validation and test sets to avoid data leakage.
- I created the model and called `fit`with the feature matrix `X_train_scaled` and the target vector `y_train`; for each observation in the validation set, the probabilities that the observation belongs to each class, as a matrix with one row per observation and one column per class. Since this is a binary classification problem, the matrix has two columns: the first contains the probability of class 0 and the second the probability of class 1. `y_pred` is a one-dimensional array containing the predicted probability that each validation observation belongs to the positive class (`diagnosis = 1`).
- If the model thinks the probability that the sample is malignant is 50% or more, I predict that the sample is malignant.
- To see whether these predictions are any good, I compared them with the actual labels of the validation set (`accuracy_score`): in about 98% of cases, the model predicted malignant or benign correctly.
- Finally I trained the model using the full feature matrix `X_full_train_scaled`and full target vector `y_full_train` and computed the predictions using the test set: accuracy on the test set is about 97%, so the model performed on unseen data about as well as on the data it was tuned with.

## Evaluation metrics for classification
- I evaluated the accuracy on the validation set for 21 decision thresholds between 0 and 1; the two extreme thresholds correspond to trivial classifiers (always class 1 and always class 0) and yield accuracies equal to the class proportions (35.1% and 65.8%); accuracy reached its maximum (99.1%) for thresholds between 0.25 and 0.45, while the default threshold of 0.5 gave 98.2%. 
Because missing a malignant tumor is the most costly error in a medical setting, I decided to adopt the lower threshold of 0.25.
- Accuracy collapses all predictions into a single number, so it cannot tell us which kinds of mistakes the model makes. `confusion matrix` summarizes the model’s predictions by comparing them with the true labels; its rows represent the actual classes and its columns the predicted ones, so that the diagonal contains the correctly classified observations (true negatives and true positives), while the off-diagonal cells contain the errors: false positives (benign cases classified as malignant) and false negatives (malignant cases classified as benign).
On the validation set, the confusion matrix shows 74 true negatives, 0 false positives, 1 false negative and 39 true positives: the model correctly identified all benign cases (`specificity` = 100%) and every case it flagged as malignant was indeed malignant (`precision` = 100%) and it missed 2 of the 40 malignant cases (`recall` = 97.5%).
- Since false negatives are more costly than false positives in a medical setting, I used the `fbeta_score`, that is a weighted harmonic mean of precision and recall that gives recall twice the importance of precision.
 On the validation set, F2-score is 0.98, because one false negative was eliminated without introducing any false positive.
- To evaluate the discriminative ability of the model independently of the decision threshold, I computed the area under the ROC curve (`roc_auc_score`) using the predicted probabilities of class 1. The ROC curve plots the true positive rate against the false positive rate for all possible thresholds. On the validation set, the model reached a ROC AUC of 0.999, indicating an almost perfect ability to discriminate between malignant and benign tumors. This is consistent with the stable accuracy observed across a wide range of decision thresholds and with the very small number of misclassified observations in the confusion matrix. Because the ROC AUC is independent of the decision threshold, it complements the threshold-dependent metrics (recall, precision, F2-score), which were used to select the final operating point of 0.25.
- Hyperparameter tuning was performed with a grid search over the regularization strength `C` of the logistic regression (values 0.01, 0.1, 1, 10 and 100), using stratified 5-fold cross-validation (`StratifiedKFold`, in order to obtain a more robust estimate of model performance) on the full training set . Standardization was included in a pipeline, so that in each fold the scaler and the model were fitted on the training folds only. For each value of C I computed the mean and standard deviation of ROC AUC, together with recall and F2-score, and selected the best configuration according to the mean ROC AUC: `C` = 1 with a mean ROC AUC of 0.995. 
- The selected model was refitted on the full training set and evaluated once on the independent test set: it achieved a ROC AUC of 0.996, an accuracy of 97.4% and an F2-score of 0.957, using a decision threshold of 0.25. The confusion matrix shows 71 true negatives, 1 false positive, 2 false negatives and 40 true positives, corresponding to a recall of 95.2% and a precision of 97.6% for the malignant class, and a specificity of 98.6%. These results are consistent with those obtained during validation and the small decrease is expected, since the decision threshold was selected on the validation set. Overall, the model discriminates very well between benign and malignant tumors; however, two malignant cases were still missed, which is the most critical type of error in a clinical setting. Given the small size of the test set, each misclassified case changes the metrics by about 1-2 percentage points, so these estimates should be interpreted with caution.


    
