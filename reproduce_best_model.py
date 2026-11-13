"""
Reproduce the results recorded in telco_churn_best_model_log.json.

Rebuilds the best model from the logged parameters, using the same data and
the same train/test split, then checks that the Test AUC and CV AUC match the
log. Exits with status 1 if the data differs or the results do not match.

Usage (from any folder):
    python reproduce_best_model.py
"""
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
LOG_PATH = HERE / 'telco_churn_best_model_log.json'
TOLERANCE = 1e-4  # the log stores AUC rounded to 4 decimals

MODEL_CLASSES = {
    'Decision Tree': DecisionTreeClassifier,
    'Lasso Logistic Regression': LogisticRegression,
    'SVM': SVC,
    'MLP Classifier': MLPClassifier,
}

failed = False


def check(label, expected, actual):
    """Print one comparison line and remember if it failed."""
    global failed
    ok = abs(expected - actual) <= TOLERANCE
    failed |= not ok
    print(f"  {label:<10} log={expected:.4f}  reproduced={actual:.4f}  {'OK' if ok else 'MISMATCH'}")


with open(LOG_PATH) as f:
    log = json.load(f)

best = log['best_model']
data_info = log['data']

# 1. Environment: a different version is a warning, not a failure
print(f"Reproducing log from {log['timestamp']}")
print("Environment:")
current_env = {
    'python': platform.python_version(),
    'numpy': np.__version__,
    'pandas': pd.__version__,
    'scikit-learn': sklearn.__version__,
}
for name, logged in log['environment'].items():
    note = '' if current_env[name] == logged else '  <-- differs, results may change slightly'
    print(f"  {name:<13} log={logged}  current={current_env[name]}{note}")

# 2. Data: must be identical, otherwise the comparison is meaningless
csv_path = HERE / data_info['file']
current_md5 = hashlib.md5(csv_path.read_bytes()).hexdigest()
if current_md5 != data_info['md5']:
    sys.exit(f"Data differs from the log (md5 {current_md5} != {data_info['md5']}). Stopping.")
print(f"Data: {data_info['file']} md5 matches")

# 3. Same preprocessing and split as the main script
df = pd.get_dummies(pd.read_csv(csv_path), drop_first=True)
X = df.drop(columns=['Churn_Yes'])
y = df['Churn_Yes']
assert df.shape[0] == data_info['n_rows_after_dummies'] and X.shape[1] == data_info['n_features']

split = data_info['split']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=split['test_size'], random_state=split['random_state'])

if best['scaled_features']:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# 4. Rebuild the best model from the logged parameters
model_class = MODEL_CLASSES[best['name']]
valid_params = model_class().get_params()
params = {k: v for k, v in best['all_params'].items()
          if k in valid_params and v != 'deprecated'}  # skip params this sklearn version no longer has
model = model_class(**params)

# 5. Cross-validated AUC (only logged when a grid search was used)
print(f"Best model: {best['name']}  params={best['best_params']}")
print("Results:")
if best['cv_auc'] is not None:
    cv_auc = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
    check('CV AUC', best['cv_auc'], cv_auc)

# 6. Fit on the training set and score on the test set
model.fit(X_train, y_train)
test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
check('Test AUC', best['test_auc'], test_auc)

print("\nReproduced successfully." if not failed else "\nResults do NOT match the log.")
sys.exit(1 if failed else 0)
