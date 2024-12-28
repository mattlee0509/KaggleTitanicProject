import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
from itertools import chain, combinations
from sklearn import svm
from sklearn.model_selection import cross_val_score, KFold


train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")


numerical_cols = ['Age','Fare']
catergorical_cols = ["Parch", "SibSp", "Pclass"]

median_train = train[numerical_cols].median()
median_test = test[numerical_cols].median()

mode_train = train[catergorical_cols].mode()
mode_test = test[catergorical_cols].mode()

train[numerical_cols] = train[numerical_cols].fillna(median_train)
test[numerical_cols] = test[numerical_cols].fillna(median_test)

train[catergorical_cols] = train[catergorical_cols].fillna(mode_train)
test[catergorical_cols] = test[catergorical_cols].fillna(mode_test)

train_dummy = pd.get_dummies(train[["Sex", "Embarked"]])
test_dummy = pd.get_dummies(test[["Sex", "Embarked"]])


train_df = pd.concat([train, train_dummy], axis=1)
test_df = pd.concat([test, test_dummy], axis=1)

train_y = train_df["Survived"].squeeze()

features = ["Age", "Pclass", "Sex_female", "Sex_male", "SibSp", "Fare", "Parch", "Embarked_C", "Embarked_Q", "Embarked_S"]


def get_subsets(features):
    return chain(*map(lambda x: combinations(features, x), range(1, len(features) + 1)))

all_subsets = list(get_subsets(features))
kf = KFold(n_splits=10, shuffle=True, random_state=0)

best_score = 0
best_subset = None
best_gamma = None


k = 2.0  
theta = 2.0  

for subset in all_subsets:
    subset_features = list(subset)
    
    train_x_subset = train_df[subset_features]
    model = svm.SVC(kernel='rbf')
    scores = cross_val_score(model, train_x_subset, train_y, cv=kf, scoring='accuracy')
    score = np.mean(scores)
    if score > best_score:
        best_score = score
        best_subset = subset_features
    # Sample multiple gamma values for each feature subset
for _ in range(30):
    best_score = 0# Sample 10 gamma values for each subset
    gamma_value = np.random.gamma(k, theta)
    train_x_subset = train_df[best_subset]
    model = svm.SVC(kernel='rbf', gamma=gamma_value)
    scores = cross_val_score(model, train_x_subset, train_y, cv=kf, scoring='accuracy')
    score = np.mean(scores)
        
    if score > best_score:
        best_score = score
        best_gamma = gamma_value

print(f"Best subset: {best_subset}")
print(f"Best gamma: {best_gamma}")
print(f"Best cross-validated accuracy: {best_score}")

# Fit the final model with the best parameters
final_model = svm.SVC(kernel='rbf', gamma=best_gamma)
final_model.fit(train_df[best_subset], train_y)

# Prepare the test data and predict
test_x_best_subset = test_df[best_subset]
test_df['Survived'] = final_model.predict(test_x_best_subset)

# Generate submission file
kaggle_submission = test_df[['PassengerId', 'Survived']]
kaggle_submission.to_csv("/kaggle/working/submission.csv", index=False)

# Print the submission dataframe
print(kaggle_submission)
