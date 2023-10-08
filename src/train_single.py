import pandas as pd

import gc

import os

from sklearn.impute import SimpleImputer

# from sklearn.preprocessing import (
#     LabelEncoder,
#     PolynomialFeatures,
#     StandardScaler,
#     OneHotEncoder,
# )

from sklearn.pipeline import make_pipeline

# from sklearn.tree import DecisionTreeClassifier, plot_tree

# from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    # RandomForestClassifier,
    # GradientBoostingClassifier,
    AdaBoostClassifier,
    # ExtraTreesClassifier,
    # VotingClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
    # BaggingClassifier,
)

from sklearn.model_selection import (
    # RandomizedSearchCV,
    train_test_split,
    # KFold,
    # cross_val_score,
)

from sklearn.metrics import roc_auc_score  # , roc_curve, auc, confusion_matrix

# from sklearn.feature_extraction.text import TfidfVectorizer

# from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval

import xgboost as xgb

# import numpy as np

# import matplotlib.pyplot as plt

# import itertools

# from tqdm import tqdm

from datetime import datetime

lab_enc = True
ohe = True
new_features = True
hp_tune = False
balance_train = False
add_tags = True
add_names = True
tfidf_title = False
oversample = False
test_count_equals_eval_count = False

TRAIN_SIZE = 0.8
TEST_SIZE = 0.2


def notify(title, text):
    os.system(
        """
              osascript -e 'display notification "{}" with title "{}"'
              """.format(
            text, title
        )
    )


comp_data = pd.read_csv("data/comp_data_pre.csv")

full_data = comp_data[comp_data["ROW_ID"].isna()]
eval_data = comp_data[comp_data["ROW_ID"].notna()]

full_data = full_data.drop(columns=["ROW_ID"])
eval_data = eval_data.drop(columns=["ROW_ID"])

# del comp_data
gc.collect()


def create_submission_file(model, feature_cols=[], filename="submission.csv"):
    global comp_data
    global eval_data

    if len(feature_cols) == 0:
        feature_cols = model.get_booster().feature_names

    # feature_cols = xgb4_model.get_booster().feature_names
    # model = final_classifier

    eval_data = comp_data[comp_data["ROW_ID"].notna()]
    # del comp_data

    # Predict on the evaluation set
    eval_data = eval_data.drop(columns=["conversion"])
    eval_data = eval_data.select_dtypes(include="number")
    y_preds = model.predict_proba(eval_data[feature_cols])[
        :, model.classes_ == 1
    ].squeeze()

    # Make the submission file
    submission_df = pd.DataFrame({"ROW_ID": eval_data["ROW_ID"], "conversion": y_preds})
    submission_df["ROW_ID"] = submission_df["ROW_ID"].astype(int)
    submission_df.to_csv(filename, index=False)


# Shuffle
full_data = full_data.sample(frac=1, random_state=19092140).reset_index(drop=True)
eval_count = eval_data.shape[0]


if test_count_equals_eval_count:
    test_data = full_data.sample(eval_count, random_state=42)
    train_data = full_data.drop(test_data.index)
else:
    train_data, test_data = train_test_split(
        full_data, test_size=TEST_SIZE, train_size=TRAIN_SIZE, random_state=42
    )


# Oversample train_data

if oversample:
    true_conversion = train_data[train_data["conversion"] == True]
    train_data = train_data.append(true_conversion).append(true_conversion)

# if NaN in conversion, drop those rows
train_data = train_data.dropna(subset=["conversion"])
test_data = test_data.dropna(subset=["conversion"])

if balance_train:

    count_converts_train = len(train_data[train_data["conversion"] == True])
    count_not_converts_train = len(train_data[train_data["conversion"] == False])

    ratio = count_converts_train / count_not_converts_train

    # From train_data, keep ratio of count_not_converts_train entries

    not_converts = train_data[train_data["conversion"] == False].sample(frac=ratio)

    new_train_data = pd.concat(
        [train_data[train_data["conversion"] == True], not_converts]
    )

    train_data = new_train_data

y_train = train_data["conversion"]
X_train = train_data.drop(columns=["conversion"])
X_train = X_train.select_dtypes(include="number")

y_test = test_data["conversion"]
X_test = test_data.drop(columns=["conversion"])
X_test = X_test.select_dtypes(include="number")

del train_data
del test_data

gc.collect()

print("Compdata shape: {}".format(comp_data.shape))
print("Full data shape: {}".format(full_data.shape))
print("Eval data shape: {}".format(eval_data.shape))

print("X_train shape: {}".format(X_train.shape))
print("X_test shape: {}".format(X_test.shape))

random_state = 12345

# abc__adaboostclassifier__learning_rate=0.2, abc__adaboostclassifier__n_estimators=50, final_estimator__colsample_bytree=0.8, final_estimator__gamma=0.1, final_estimator__learning_rate=0.1, final_estimator__max_depth=10, final_estimator__min_child_weight=3, final_estimator__n_estimators=25, final_estimator__subsample=0.7, hgb__l2_regularization=0.3, hgb__learning_rate=0.1, hgb__max_depth=4, hgb__max_iter=500, hgb__max_leaf_nodes=20, hgb__min_samples_leaf=40, xgb__colsample_bytree=0.6, xgb__gamma=0.1, xgb__learning_rate=0.1, xgb__max_depth=100, xgb__min_child_weight=3, xgb__n_estimators=25, xgb__subsample=0.7;, score=0.881

best_params_abc = {
    "learning_rate": 0.1,
    "n_estimators": 25,
}

best_params_xgb = {
    "colsample_bytree": 0.9,
    "gamma": 0.2,
    "learning_rate": 0.2,
    "max_depth": 10,
    "min_child_weight": 0,
    "n_estimators": 100,
    "subsample": 0.7,
}

best_params_hgb = {
    "l2_regularization": 0.4,
    "learning_rate": 0.3,
    "max_depth": 7,
    "max_iter": 400,
    "max_leaf_nodes": 30,
    "min_samples_leaf": 20,
}

best_params_final = {
    "colsample_bytree": 0.6,
    "gamma": 0.5,
    "learning_rate": 0.01,
    "max_depth": 5,
    "min_child_weight": 3,
    "n_estimators": 50,
    "subsample": 0.6,
}

model = StackingClassifier(
    [
        (
            "xgb",
            xgb.XGBClassifier(
                **best_params_xgb,
                objective="binary:logistic",
                seed=random_state,
                n_jobs=-1,
                verbose=20,
            ),
        ),
        ("hgb", HistGradientBoostingClassifier(**best_params_hgb)),
        (
            "abc",
            make_pipeline(
                SimpleImputer(strategy="median"), AdaBoostClassifier(**best_params_abc)
            ),
        ),
    ],
    final_estimator=xgb.XGBClassifier(
        **best_params_final,
        objective="binary:logistic",
        seed=random_state,
        n_jobs=-1,
        verbose=20,
    ),
    n_jobs=-1,
    stack_method="predict_proba",
    verbose=20,
)

print("Fitting...")

model.fit(X_train, y_train)

score = roc_auc_score(y_test, model.predict_proba(X_test)[:, model.classes_ == 1])

print("Done. Score: {}".format(score))

notify("STACK OPTIMIZED", "Done. Score: {}".format(score))

print("Creating submission file...")

create_submission_file(
    model,
    model.named_estimators_["xgb"].get_booster().feature_names,
    "outputs/submission_stack_optimized(xgb, hgb, abc)_polyfeat_imp_pred_pdp_pca_embs: {:.5f} {}.csv".format(
        score, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ),
)

print("Fitting on full data...")

model.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))

print("Creating submission file...")

create_submission_file(
    model,
    model.named_estimators_["xgb"].get_booster().feature_names,
    "outputs/submission_stack_optimized_full(xgb, hgb, abc)_polyfeat_imp_pred_pdp_pca_embs: {:.5f} {}.csv".format(
        score, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ),
)

print("Done.")
