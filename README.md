# Telco Churn Profit Optimization

Predict which telecom customers are likely to churn, then decide **how many of them to send a retention offer to** so that campaign profit is maximized.

The retention offer costs $200 per customer and always convinces the customer to renew for another year. Budget covers at most 25% of expiring contracts, so the question is not only *who* will churn, but *how far down the ranked list it is still profitable to go*.

## Approach

1. **Prepare data.** Dummy-encode the categorical features (`get_dummies`, `drop_first=True`) and split 80/20 (`random_state=123`).
2. **Explore tenure vs. churn.** Churn drops sharply in the first months and keeps falling with tenure, a non-linear effect (`churn_vs_tenure.png`).
3. **Compare four models** with one shared `evaluate_model` function (optional scaling, optional `GridSearchCV` scored by AUC, fit, predict, ROC):
   - Decision Tree (grid search over `max_depth`, `min_samples_leaf`)
   - Lasso Logistic Regression (grid search over `C`)
   - SVM (linear kernel, default hyperparameters)
   - MLP Classifier (default-style setup, no tuning)
4. **Pick the best model** by Test AUC.
5. **Cost/benefit matrix.** A saved customer is worth a year of revenue minus the offer cost; an unnecessary offer is a $200 loss.
6. **Profit curve.** Rank customers by churn probability, target the top *k*, and compute total profit for every *k*.
7. **Extra credit.** Use each customer's own `MonthlyCharges` to compute an individualized expected value instead of a single average.

## Results

Test set of 1,407 customers.

| Model | Test AUC |
|---|---|
| **Lasso Logistic Regression** (`C=0.1`) | **0.8592** |
| SVM | 0.8554 |
| Decision Tree | 0.8427 |
| MLP Classifier | 0.7603 |

Cross-validated AUC of the best model: 0.8398.

**Cost/benefit** (average monthly charge $64.80, so $777.58 per year): a saved customer is worth +$577.58 net, an unnecessary offer costs -$200.

| Strategy | Customers targeted | Profit (test set) |
|---|---|---|
| Budget cap (top 25%) | 351 | $117,974 |
| Optimal cutoff (average revenue) | 657 (46.7%) | **$143,085** |
| Optimal cutoff (individual expected value) | 608 (43.2%) | **$201,564** |

Raising the target from 25% to about 47% adds roughly $25,000 on the test set. Using each customer's own monthly charges adds another $58,000.

**Model vs. random targeting.** Marketing to the same 657 customers chosen at random (average of 1,000 simulations) earns about $17,853 on the test set, versus $143,085 for the model. Scaled to 100,000 expiring contracts, that is about $1.27M vs. $10.17M, an incremental value of roughly $8.9M.

## Project files

| File | Purpose |
|---|---|
| `telco_churn_profit_optimization.py` | Full analysis: models, cost/benefit, profit curve |
| `reproduce_best_model.py` | Rebuilds the best model from the log and checks the AUC matches |
| `telco_churn.csv` | Input data: 7,032 customers, 18 features plus `Churn` |
| `telco_churn_best_model_log.json` | Best model, parameters, AUCs, data hash and library versions |
| `churn_vs_tenure.png` | Tenure vs. churn plot |

## Run it

```bash
pip install numpy pandas matplotlib seaborn scikit-learn ipython
python telco_churn_profit_optimization.py
```

Run it from this folder, because the script reads `telco_churn.csv` and writes its outputs using relative paths.

## Reproduce the logged result

```bash
python reproduce_best_model.py
```

It checks the CSV hash, rebuilds the best model from the logged parameters on the same split, and confirms the CV and Test AUC match the log. The log was produced with Python 3.10.10, numpy 1.26.4, pandas 2.2.2 and scikit-learn 1.7.2; other versions may give slightly different numbers.
