"""
This file was initially written on Jupyter Notebook for convenience developing
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import squarify
import matplotlib.dates as dates
from datetime import datetime
import seaborn as sns
from random import sample
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from IPython.display import Image
from pydotplus import graph_from_dot_data


# covert utc timestamp string to date yyyy-mm-dd format in datetime object
def time_to_date(utc_timestamp):
    date = datetime.strptime(utc_timestamp[0:10], '%Y-%m-%d').date()
    return date


# data understanding
data = pd.read_csv('2019-Dec.csv')
print(data.shape, data.info(), data.columns)

# count of event_type
print(data['event_type'].value_counts())

# purchase daily trend
data['event_date'] = data['event_time'].apply(lambda x: time_to_date(x))
purchase = data.loc[data['event_type'] == 'purchase']
purchase_by_date = purchase[['event_date', 'event_type']].groupby(['event_date']).count()
x = pd.Series(purchase_by_date.index.values)
y = purchase_by_date['event_type']
plt.rcParams['figure.figsize'] = (20, 8)

plt.plot(x, y)
plt.show()

# price daily trend
product_price_trend = data[['event_date','price']].groupby(['event_date']).mean()
print(product_price_trend)
x = pd.Series(product_price_trend.index.values)
y = product_price_trend['price']
plt.rcParams['figure.figsize'] = (20,8)

plt.plot(x,y)
plt.show()
# purchase behavior
labels = ['view', 'cart', 'purchase', 'remove_from_cart']
size = data['event_type'].value_counts()
explode = [0, 0.1, 0.1, 0.1]

plt.rcParams['figure.figsize'] = (8, 8)
plt.pie(size, explode=explode, labels=labels, shadow=True, autopct='%.2f%%')
plt.title('Event_Type', fontsize=20)
plt.axis('off')
plt.legend()
plt.show()

# Predict if user will purchase a given product after adding to cart?
print(data.loc[data['event_type'] == 'purchase'])
# count null values in the table
print(data.apply(lambda ss: sum(ss.isnull()), axis=0))
clean_data = data.dropna(subset=['brand'])
clean_data = clean_data.dropna(subset=['user_session'])
clean_data.drop('category_code', axis=1, inplace=True)
# count null values in the cleaned table
print(clean_data.apply(lambda xx: sum(xx.isnull()), axis=0))
# construct the data frame for model using
product = clean_data[['product_id', 'price']]
product = product.drop_duplicates(subset=['product_id'])
cart_count = data.loc[data['event_type'] == 'cart'].groupby('product_id').size().reset_index(name='cart_count')
view_count = data.loc[data['event_type'] == 'view'].groupby('product_id').size().reset_index(name='view_count')
purchase_count = data.loc[data['event_type'] == 'purchase'].groupby('product_id').size().reset_index(
    name='purchase_count')
model_table = product.join(cart_count.set_index('product_id'), on='product_id').join(view_count.set_index('product_id'),
                                                                                     on='product_id').join(
    purchase_count.set_index('product_id'), on='product_id').fillna(0)
clean_data['event_date'] = clean_data['event_time'].apply(lambda s: time_to_date(s))
df = clean_data.loc[data["event_type"].isin(["cart", "purchase"])].drop_duplicates(
    subset=['event_type', 'product_id', 'price', 'user_id',
            'user_session'])
df["is_purchased"] = np.where(df["event_type"] == "purchase", 1, 0)
df["is_purchased"] = df.groupby(["user_session", "product_id"])["is_purchased"].transform("max")
df = df.loc[df["event_type"] == "cart"].drop_duplicates(["user_session", "product_id", "is_purchased"])
df['event_weekday'] = df['event_date'].apply(lambda s: s.weekday())
new_df = df[['product_id', 'is_purchased', 'event_weekday']]
df = new_df.join(model_table.set_index('product_id'), on='product_id').fillna(0)

# Split into test and training sets
# split the data into test/training/validation sets
train, test, validate = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])

# blow part is reference from https://lukesingham.com/whos-going-to-leave-next/
# separate target and predictors
y_train = train['is_purchased']
x_train = train.drop(['is_purchased'], axis=1)
y_test = test['is_purchased']
x_test = test.drop(['is_purchased'], axis=1)
y_validate = validate['is_purchased']
x_validate = validate.drop(['is_purchased'], axis=1)
# check the sets
print(train.shape, test.shape, validate.shape, y_test.mean(), y_train.mean())
# Variable importance
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), x_train), reverse=True)

# Decision Tree
# Instantiate with a max depth of 3/4/5
tree_model = tree.DecisionTreeClassifier(max_depth=3)
# Fit a decision tree
tree_model = tree_model.fit(x_train, y_train)
# Training accuracy
tree_model.score(x_train, y_train)

# Predictions/probs on the test dataset
predicted = pd.DataFrame(tree_model.predict(x_test))
probs = pd.DataFrame(tree_model.predict_proba(x_test))

# Store metrics
tree_accuracy = metrics.accuracy_score(y_test, predicted)
tree_roc_auc = metrics.roc_auc_score(y_test, probs[1])
tree_confus_matrix = metrics.confusion_matrix(y_test, predicted)
tree_classification_report = metrics.classification_report(y_test, predicted)
tree_precision = metrics.precision_score(y_test, predicted, pos_label=1)
tree_recall = metrics.recall_score(y_test, predicted, pos_label=1)
tree_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

# evaluate the model using 10-fold cross-validation
# tree_cv_scores = cross_val_score(tree.DecisionTreeClassifier(max_depth=3), x_test, y_test, scoring='precision', cv=10)

# output decision plot
dot_data = tree.export_graphviz(tree_model, out_file=None,
                                feature_names=x_test.columns.tolist(),
                                class_names=['Not_purchase', 'Purchase'],
                                filled=True, rounded=True,
                                special_characters=True)
graph = graph_from_dot_data(dot_data)
graph.write_png("decision_tree.png")

# Random Forest
# Instantiate
rf = RandomForestClassifier()
rf_model = rf.fit(x_train, y_train)
rf_model.score(x_train, y_train)

# Predictions/probs on the test dataset
predicted = pd.DataFrame(rf_model.predict(x_test))
probs = pd.DataFrame(rf_model.predict_proba(x_test))

# Store metrics
rf_accuracy = metrics.accuracy_score(y_test, predicted)
rf_roc_auc = metrics.roc_auc_score(y_test, probs[1])
rf_confus_matrix = metrics.confusion_matrix(y_test, predicted)
rf_classification_report = metrics.classification_report(y_test, predicted)
rf_precision = metrics.precision_score(y_test, predicted, pos_label=1)
rf_recall = metrics.recall_score(y_test, predicted, pos_label=1)
rf_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

# KKN
# instantiate learning model (k = 3)
knn_model = KNeighborsClassifier(n_neighbors=30)
# fit the model
knn_model.fit(x_train, y_train)
# Accuracy
knn_model.score(x_train, y_train)

# Predictions/probs on the test dataset
predicted = pd.DataFrame(knn_model.predict(x_test))
probs = pd.DataFrame(knn_model.predict_proba(x_test))

# Store metrics
knn_accuracy = metrics.accuracy_score(y_test, predicted)
knn_roc_auc = metrics.roc_auc_score(y_test, probs[1])
knn_confus_matrix = metrics.confusion_matrix(y_test, predicted)
knn_classification_report = metrics.classification_report(y_test, predicted)
knn_precision = metrics.precision_score(y_test, predicted, pos_label=1)
knn_recall = metrics.recall_score(y_test, predicted, pos_label=1)
knn_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

# GasussianNB
# Instantiate
bayes_model = GaussianNB()
# Fit the model
bayes_model.fit(x_train, y_train)
# Accuracy
bayes_model.score(x_train, y_train)

# Predictions/probs on the test dataset
predicted = pd.DataFrame(bayes_model.predict(x_test))
probs = pd.DataFrame(bayes_model.predict_proba(x_test))

# Store metrics
bayes_accuracy = metrics.accuracy_score(y_test, predicted)
bayes_roc_auc = metrics.roc_auc_score(y_test, probs[1])
bayes_confus_matrix = metrics.confusion_matrix(y_test, predicted)
bayes_classification_report = metrics.classification_report(y_test, predicted)
bayes_precision = metrics.precision_score(y_test, predicted, pos_label=1)
bayes_recall = metrics.recall_score(y_test, predicted, pos_label=1)
bayes_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

# Result of Models
# Model comparison KNN = 30
models = pd.DataFrame({
  'Model': ['d.Tree', 'r.f.', 'kNN',  'Bayes'],
  'Accuracy': [tree_accuracy, rf_accuracy, knn_accuracy, bayes_accuracy],
  'Precision': [tree_precision, rf_precision, knn_precision, bayes_precision],
  'recall': [tree_recall, rf_recall, knn_recall, bayes_recall],
  'F1': [tree_f1, rf_f1, knn_f1, bayes_f1]
})
# Print table and sort by test precision
models.sort_values(by='Precision', ascending=False)
