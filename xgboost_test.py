from fileinput import filename
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

df = pd.read_csv('final_table_13-04-2022.csv', dtype={'Calendar Year': str, 'CDP Planning Year': str, 'Calendar Year/Month': str, 'Sales Plant (central)': str, 'Final Customer Group': str, 'Product Hierarchy': str, 'Additional Attributes': str})
#df = df[18688:]
df.drop(df.index[:18688],axis=0, inplace=True)
df.drop(['Additional Attributes'], axis=1, inplace=True)
print(df.head())

#eliminate whitespace in column names:
df.columns = df.columns.str.replace(' ', '_')
print(df.head())

print(df.dtypes)


#split into X & Y
X = df.drop('Label', axis=1).copy()
print(X.head())
y = df['Label'].copy()
print(y.head())

#formatting X with One-Hot Encoding: (--> für Datum sinnvoll oder nicht?!)
X_encoded = pd.get_dummies(X,columns=['Calendar_Year', 'CDP_Planning_Year', 'Calendar_Year/Month','Sales_Plant_(central)','Final_Customer_Group','Product_Hierarchy'])
print(X_encoded.head())

print(y.mean())

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42)

print(y_train.mean())
print(y_test.mean())

#build model
clf_xgb = xgb.XGBClassifier(seed=42, n_estimators=1)
clf_xgb.fit(X_train, y_train, verbose=True)

bst = clf_xgb.get_booster()
for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
    print('%s: ' % importance_type, bst.get_score(importance_type=importance_type))

node_params = {'shape': 'box', 'style': 'filled, rounded', 'fillcolor': '#78cbe'}
leaf_params = {'shape': 'box', 'style': 'filled', 'fillcolor': '#e48038'}

graph_data = xgb.to_graphviz(clf_xgb, num_trees=0, size="10,10", condition_node_params=node_params, leaf_node_params=leaf_params)

print(graph_data)

graph_data.view(filename='xgboost_tree')

