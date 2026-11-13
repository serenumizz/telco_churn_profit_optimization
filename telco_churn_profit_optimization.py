"""
List of the data
- Gender: Whether the customer is a male or a female
- SeniorCitizen: Whether the customer is a senior citizen or not (1, 0)
- Partner: Whether the customer has a partner or not (Yes, No)
- Dependents: Whether the customer has dependents or not (Yes, No)
- Tenure: Number of months the customer has stayed with the company
- PhoneService: Whether the customer has a phone service or not (Yes, No)
- MultipleLines: Whether the customer has multiple lines or not (Yes, No, No phone service)
- InternetService: Customer’s internet service provider (DSL, Fiber optic, No)
- OnlineSecurity: Whether the customer has online security or not (Yes, No, No internet service)
- OnlineBackup: Whether the customer has online backup or not (Yes, No, No internet service)
- DeviceProtection: Whether the customer has device protection or not (Yes, No, No internet service)
- TechSupport: Whether the customer has tech support or not (Yes, No, No internet service)
- StreamingTV: Whether the customer has streaming TV or not (Yes, No, No internet service)
- StreamingMovies: Whether the customer has streaming movies or not (Yes, No, No internet service)
- Contract: The contract term of the customer (Month-to-month, One year, Two year)
- PaperlessBilling: Whether the customer has paperless billing or not (Yes, No)
- PaymentMethod: The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
- MonthlyCharges: The amount charged to the customer monthly
- Churn: Whether the customer churned or not shortly after contract expiration (Yes or No)
"""

#load important packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve


def evaluate_model(model, X_train, X_test, y_train, y_test, param_grid=None, scale=False, cv=5):
    """
    Evaluate a classification model using ROC AUC score and classification report.

    Parameters:
    - model: The classification model to evaluate.
    - X_test: Test features.
    - y_test: True labels for the test set.
    - param_grid: Optional parameter grid for hyperparameter tuning (default is None).
    - scale: Boolean indicating whether to scale features (default is False).
    - cv: Number of cross-validation folds (default is 5).

    Returns:
    - auc_score: ROC AUC score of the model on the test set.
    - report: Classification report as a string.
    """

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if param_grid:
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='roc_auc'
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_
    else:
        model.fit(X_train, y_train)
        best_params = None
        best_score = None

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return {'model': model, 'best_params': best_params, 'best_score': best_score, 
            'y_prob': y_prob, 'y_pred': y_pred, 'fpr': fpr, 'tpr': tpr, 'auc': roc_auc_score(y_test, y_prob)}




# 1. Load the churn data (telco_churn.csv) and use get_dummies to transform
# the categorical features into dummy features (drop_first=True).

df = pd.read_csv('telco_churn.csv')
df = pd.DataFrame(df)

df = pd.get_dummies(df, drop_first=True)

# 2. Explore the relationship between Tenure and Churn
# Detailed exploratory work is skipped because data quality is high and there are no missing values.
# Tenure is often a key churn feature, so we plot the average churn rate at each tenure value
# (blue line) against the overall average churn rate (red dashed line) to see whether the
# relationship is linear or non-linear.

# Calculate average churn rate by tenure
tenure_churn = df.groupby('tenure')['Churn_Yes'].mean().reset_index()

# Overall baseline average churn
overall_churn_rate = df['Churn_Yes'].mean()

plt.figure(figsize=(10, 6))
# Line 1: Churn rate by tenure
plt.plot(tenure_churn['tenure'], tenure_churn['Churn_Yes'], label='Churn Rate by Tenure', color='blue', linewidth=2)
# Line 2: Overall average churn baseline
plt.axhline(y=overall_churn_rate, color='red', linestyle='--', label=f'Overall Average Churn ({overall_churn_rate:.1%})', linewidth=2)

plt.title('Churn Probability vs. Tenure (Double Line Chart)')
plt.xlabel('Tenure (months)')
plt.ylabel('Churn Probability / Rate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('churn_vs_tenure.png', dpi=300, bbox_inches='tight')
plt.close()

"""The plot depicts the correlation between tenure and churn probability which has negative relationship. 
The longer tenure the customer has, the lower probability of churn they have. 
However, the steepest drop-off is even before the average churn, it is between 0-8 months after cuctomer join. 
Furthermore, that most of customer which stays longer than 30 months (over 60 months) become highly loyal customer."""

# 3. Define X and y and split the data into Training and Test using an 80/20 split
# (random_state is set for reproducibility). Then build the best decision tree classifier
# on the TRAINING set using GridSearchCV over:
#   - max_depth within the range [2,15]
#   - min_samples_leaf within the range [1,100] (in steps of 10)
# with 5-fold CV and AUC ("roc_auc") as the scoring function, to find the optimal
# values of min_samples_leaf and max_depth.

X = df.drop(columns=['Churn_Yes'])
y = df['Churn_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# Define the Decision Tree model and the parameter grid for GridSearchCV

tree = evaluate_model(DecisionTreeClassifier(random_state=123),
                      X_train, X_test, y_train, y_test,
                      param_grid={'max_depth': range(2, 15),
                                  'min_samples_leaf': range(1, 100, 10)})
print("---- Decision Tree Model Evaluation ----")
print(f"Best Parameters: {tree['best_params']}")
print(f"Test AUC: {tree['auc']:.4f}")

# # 4. Calculate the feature importances of the best decision tree. The top 5 most important
# # features are then explained as positively or negatively correlated with churn.

tree_model = tree['model']  # already the best estimator, fitted inside evaluate_model

importances = tree_model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("---- Feature Importances from Decision Tree ----")
print(feature_importance_df.head(5))
print(classification_report(y_test, tree['y_pred']))


# Explanation of the top 5 features based on their importance and correlation with churn:
# """1. Tenure: Negative
#    - As analyzed in our earlier visualization, customers with longer tenure are much less likely to churn. Loyal customers build habits, trust, and might have bundled services that make leaving the company less attractive.

# 2. InternetService_Fiber optic : Positive
#    - The variable indicates whether the customer has Fiber Optic internet. Fiber Optic customers show a higher churn and it might point to service reliability issues, high technical support needs, or intense competitor pricing/promotions specifically targeting fiber customers.

# 3. Monthly Charges: Positive
#    - Higher monthly charges are associated with a higher of churn. Customers paying premium rates are more price-sensitive and active in seeking out cheaper alternatives, discounts, or competitors.

# 4. StreamingMovies_No internet service: Negative
#    - This category represents customers who do not have internet service at all (and thus no streaming movies). These customers have a significantly lower churn probability because they typically use basic, stable analog services (like traditional landlines) which have lower churn rates compared to high-speed broadband products.

# 5. Contract_Two year: Negative
#    - Customers locked into long-term contract structures (Two-year agreements) have a much lower probability of churn because they face early termination penalties and have committed to staying with the brand long-term."""



# 5. Fit Lasso Regression (LogisticRegression with penalty='l1') to the Training set and use
# GridSearchCV to find the best value of the hyperparameter C in [0.01, 0.1, 0.5, 1, 5, 10, 50, 100].
# Report the optimal value of C and define lr_best that uses it.


lr = evaluate_model(LogisticRegression(penalty='l1', solver='liblinear', random_state=123),
                    X_train, X_test, y_train, y_test,
                    param_grid={'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100]},
                    scale=True)

print("\n\n---- Lasso Regression Model Evaluation ----")
print(f"Best Parameters: {lr['best_params']}")
print(f"Test AUC: {lr['auc']:.4f}")
print("---- Feature Importances from LR ----")
print(feature_importance_df.head(5))

# 6. Print the regularized coefficient values in lr_best (4 decimal places) to see how many of
# the Lasso coefficients have shrunken all the way to zero, then interpret them.

print(f"Best cross-validation AUC: {lr['best_score']:.4f}")
print("Regularized Coefficients:")
for feature, coef in zip(X_train.columns, lr['model'].coef_[0]):
    if abs(coef) == 0:
        print(f"{feature}: {coef:.4f}")
print(classification_report(y_test, lr['y_pred']))

# """Out of the 29 features in our encoded dataset, only 1 coefficient has shrunken all the way to exactly 0.0000:
# `MultipleLines_No phone service: 0.0000`

# Lasso (L1 regularization) uses automatic feature selection by zeroing out coefficients of features that add no predictive value beyond what is already captured by other variables. MultipleLines_No phone service is redundant because the information is already captured by PhoneService_Yes .

# InternetService_Fiber optic (1.1208): Having fiber optic internet is the strongest indicator that a customer is likely to churn. This continues to support our hypothesis that there may be quality, reliability, or high pricing issues associated with the fiber optic packages.
# Other strong positive drivers include StreamingTV_Yes, StreamingMovies_Yes, and MultipleLines_Yes.

# Strongest Retention Drivers (Negative Impact on Churn):
# Contract_Two year & Contract_One year : Long-term contracts are effective barriers to churn. A two-year contract has the strongest negative association with churn of any feature in the entire model.
# StreamingMovies_No internet service : Customers without internet service at all are stable and very unlikely to leave.
# OnlineSecurity_Ye and TechSupport_Yes : Add-on security and support features significantly lower the likelihood of churn, showing that engagement with utility/support features builds loyalty."""

# # 7. Pick a third classification model (SVM), fit it with default hyperparameters
# # on the scaled features, and see how well it does (AUC).

# # SVM
svm = evaluate_model(SVC(kernel='linear', probability=True, random_state=123),
                     X_train, X_test, y_train, y_test, param_grid={'C': [0.1, 1, 10]}, scale=True)

print("\n\n---- SVM Model Evaluation ----")
print(f"Best Parameters: {svm['best_params']}")
print(f"Test AUC: {svm['auc']:.4f}")
print(f"Probability: {svm['y_prob'][:5]}")
print(f"Prediction: {svm['y_pred'][:5]}")
print(classification_report(y_test, svm['y_pred']))

# MLP Classifier


# Initialize SVM with probability=True so we can compute ROC AUC later
mlp = evaluate_model(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
                     X_train, X_test, y_train, y_test, param_grid=None, scale=True)

# Predict probabilities and classes
y_prob = mlp['y_prob']
y_pred = mlp['y_pred']

# Calculate AUC
print(f"MLP Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"MLP Test AUC: {mlp['auc']:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(classification_report(y_test, mlp['y_pred']))

# 8. Review the AUC values to identify the best of the three and call it best_model.

# 4. the best model based on the highest test set AUC
models_auc = {
    'Decision Tree': (tree, tree['auc']),
    'Lasso Logistic Regression': (lr, lr['auc']),
    'SVM': (svm, svm['auc']),
    'MLP Classifier': (mlp, mlp['auc'])
}

best_model_name, (best_model_obj, best_auc) = max(models_auc.items(), key=lambda item: item[1][1])

print(f"The best performing model on the TEST set is: {best_model_name} with an AUC of {best_auc:.4f}")

# Re-assign the overall `best_model` variable to this optimal model
best_model = best_model_obj

# Log the best model and everything needed to reproduce the result
import json, hashlib, platform
from datetime import datetime
import sklearn

with open('telco_churn.csv', 'rb') as f:
    data_md5 = hashlib.md5(f.read()).hexdigest()

run_log = {
    'timestamp': datetime.now().isoformat(timespec='seconds'),
    'best_model': {
        'name': best_model_name,
        'test_auc': round(float(best_auc), 4),
        'cv_auc': None if best_model_obj['best_score'] is None else round(float(best_model_obj['best_score']), 4),
        'best_params': best_model_obj['best_params'],
        'all_params': best_model_obj['model'].get_params(),
        'scaled_features': best_model_name != 'Decision Tree',
    },
    'all_models_test_auc': {name: round(float(v[1]), 4) for name, v in models_auc.items()},
    'data': {
        'file': 'telco_churn.csv',
        'md5': data_md5,
        'n_rows_after_dummies': int(df.shape[0]),
        'n_features': int(X.shape[1]),
        'split': {'test_size': 0.2, 'random_state': 123},
    },
    'environment': {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'scikit-learn': sklearn.__version__,
    },
}

with open('telco_churn_best_model_log.json', 'w') as f:
    json.dump(run_log, f, indent=2, default=str)
print("Saved best model log to telco_churn_best_model_log.json")

# 9. Define the cost / benefit matrix for implementing this model. The cost of the offer is $200
# and everyone who receives it takes it. The benefit comes when the offer goes to a customer whose
# actual Churn value is 1 (they would have left but were saved). 
# Each customer is assumed to generate the same revenue, equal to the average MonthlyCharges, reported as a YEARLY value.

# Calculate average MonthlyCharges across the dataset
avg_monthly_charges = df['MonthlyCharges'].mean()
avg_yearly_revenue = avg_monthly_charges * 12

# Cost/Benefit metrics
cost_per_offer = 200
tp_net_benefit = avg_yearly_revenue - cost_per_offer
fp_net_cost = -cost_per_offer

print(f"\n\n--- Cost/Benefit Matrix ---")
print(f"Assumes cost per offer: ${cost_per_offer:.2f}")
print(f"Average Monthly Charges: ${avg_monthly_charges:.2f}")
print(f"Average Yearly Revenue per Customer: ${avg_yearly_revenue:.2f}")
print(f"Cost per Target Offer: ${cost_per_offer:.2f}")
print(f"True Positive Net Benefit (Yearly): ${tp_net_benefit:.2f}")
print(f"False Positive Net Cost: ${fp_net_cost:.2f}")

# 10. Using the best model along with the cost/benefit matrix, plot a PROFIT CURVE that shows the
# overall profit at different levels of marketing. Sort customers by predicted probability, then
# calculate the total profit at each threshold by determining who is marketed to and who is not.

# Calculate expected baseline numbers
sorted_indices = np.argsort(lr['y_prob'])[::-1] # sort by predicted probability in descending 
y_prob_sorted = lr['y_prob'][sorted_indices]
y_test_sorted = y_test.iloc[sorted_indices].values

# Values from our cost/benefit matrix
tp_benefit = tp_net_benefit
fp_cost = fp_net_cost

# Lists to store metrics for plotting
percentages_targeted = []
profits = []
thresholds = []

n_total = len(y_test_sorted)

# Loop through all possible target amounts (from 0 to 100% of the test set)
for k in range(n_total + 1):
    # If we target the top k customers with the highest predicted probability
    targeted_y = y_test_sorted[:k]

    # Count TP and FP within the targeted group
    tp = np.sum(targeted_y == True)
    fp = np.sum(targeted_y == False)

    # Compute profit
    total_profit = (tp * tp_benefit) + (fp * fp_cost)

    pct_targeted = (k / n_total) * 100
    threshold = y_prob_sorted[k-1] if k > 0 else 1.0

    percentages_targeted.append(pct_targeted)
    profits.append(total_profit)
    thresholds.append(threshold)

# Find maximum profit and corresponding percentage targeted
max_profit = max(profits)
max_idx = profits.index(max_profit)
optimal_pct = percentages_targeted[max_idx]
optimal_threshold = thresholds[max_idx]

# Find profit at Nadia's 25% target budget limit
nadia_target_k = int(0.25 * n_total)
nadia_profit = profits[nadia_target_k]
nadia_pct = percentages_targeted[nadia_target_k]
nadia_threshold = thresholds[nadia_target_k]

print(f"\n\n--- Maximum profit and corresponding percentage targeted ---")
print(f"Key Metrics on the Test Set ({n_total} Customers) ---")
print(f"Nadia's 25% Target (Top {nadia_target_k} customers):")
print(f"Profit: ${nadia_profit:,.2f}")
print(f"Threshold Probability: {nadia_threshold:.4f}")
print(f"Optimal Point (Top {max_idx} customers / {optimal_pct:.2f}% of base):")
print(f"Max Profit: ${max_profit:,.2f}")
print(f"Optimal Threshold Probability: {optimal_threshold:.4f}")
print(f"Incremental profit over Nadia's 25% cap: ${max_profit - nadia_profit:,.2f}")

# # 11. Estimate the amount of money made for the company by using the model, and compare its profit against
# # a "dumb" model that markets to the same number of customers in the Test set selected randomly
# # (simulated with a random number generator). Scale the results up to 100,000 customers.

# Set random seed for reproducibility in random marketing simulation
np.random.seed(123)

# Target parameters
target_k = max_idx # number of targeted customers under optimal model (top-k with max profit)
tp_benefit = tp_net_benefit
fp_cost = fp_net_cost
n_test = len(y_test)

# Simulate a random 'dumb' campaign targeting the same number (target_k) of random customers from the test set
# We run this simulation multiple times to get a stable estimate of random performance
simulations = 1000
random_profits = []

for _ in range(simulations):
    # Randomly pick indices to market to
    randomly_targeted_indices = np.random.choice(y_test.index, size=target_k, replace=False)
    targeted_actual_churn = y_test.loc[randomly_targeted_indices]

    tp_random = np.sum(targeted_actual_churn == True)
    fp_random = np.sum(targeted_actual_churn == False)

    random_profit = (tp_random * tp_benefit) + (fp_random * fp_cost)
    random_profits.append(random_profit)

avg_random_profit = np.mean(random_profits)

# Net value made on the test set
net_test_value_added = max_profit - avg_random_profit

# Scale up to a base of 100,000 expiring contracts
scaling_factor = 100000 / n_test
scaled_model_profit = max_profit * scaling_factor
scaled_random_profit = avg_random_profit * scaling_factor
scaled_incremental_value = net_test_value_added * scaling_factor

print(f"RESULTS FOR THE TEST SET ({n_test} customers) ---")
print(f"Optimized Model Profit (Targeting top {target_k}): ${max_profit:,.2f}")
print(f"Average 'Dumb' Random Campaign Profit: ${avg_random_profit:,.2f}")
print(f"Net Value Generated (Test Set): ${net_test_value_added:,.2f}")
print("\nSCALED TO 100,000 EXPIRING CONTRACTS ---")
print(f"Scaled Optimized Model Profit: ${scaled_model_profit:,.2f}")
print(f"Scaled Random Campaign Profit: ${scaled_random_profit:,.2f}")
print(f"INCREMENTAL VALUE MADE FOR THE COMPANY: ${scaled_incremental_value:,.2f}")

# # 13. Calculate an individualized expected value (EV) of profit/loss
# # for each customer using their specific MonthlyCharges and predicted churn probability, use the sorted
# # EV to decide whom to market to, calculate the profit, and compare whether the results are any better.

# 1. Compute individualized Expected Value (EV) for each customer in the test set
test_monthly_charges = X_test['MonthlyCharges'].values
individual_ev = lr['y_prob'] * (12 * test_monthly_charges) - 200

# Create a DataFrame to help with sorting and actual outcome matching
evaluations_df = pd.DataFrame({
    'actual_churn': y_test.values,
    'monthly_charges': test_monthly_charges,
    'ev': individual_ev
})

# 2. Sort customers by their expected value in descending order
evaluations_df_sorted = evaluations_df.sort_values(by='ev', ascending=False).reset_index(drop=True)

# 3. Calculate cumulative actual profit as we target down the sorted list
cumulative_profits_ind = []
cumulative_targeted_pct = []
n_total_ind = len(evaluations_df_sorted)

for k in range(n_total_ind + 1):
    targeted_sub = evaluations_df_sorted.iloc[:k]
    actual_tps = targeted_sub[targeted_sub['actual_churn'] == True]
    actual_fps = targeted_sub[targeted_sub['actual_churn'] == False]

    # TP actual profit: (12 * MonthlyCharges) - 200
    tp_profit = (12 * actual_tps['monthly_charges'] - 200).sum()
    # FP actual loss: -200
    fp_loss = -200 * len(actual_fps)

    net_profit = tp_profit + fp_loss
    cumulative_profits_ind.append(net_profit)
    cumulative_targeted_pct.append((k / n_total_ind) * 100)

# Find maximum profit with individualized pricing
max_profit_ind = max(cumulative_profits_ind)
max_idx_ind = cumulative_profits_ind.index(max_profit_ind)
optimal_pct_ind = cumulative_targeted_pct[max_idx_ind]

print(f"\n\n--- COMPARISON ---")
print("Individualized expected value (EV) of profit/loss")
print(f"Maximum Profit using Average Monthly Charges: ${max_profit:,.2f}")
print(f"Maximum Profit using Individualized expected value: ${max_profit_ind:,.2f}")
print(f"Incremental Profit gain using individualized EV: ${max_profit_ind - max_profit:,.2f}")
print(f"Optimal target size: {optimal_pct_ind:.2f}% (Top {max_idx_ind} customers)")