#Importing Libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier

#Reading data
data = pd.read_csv(r"Social_Network_Ads.csv")
data.head()

#Removing User ID, as it's an increamental value that doesn't add to our classification prediction
data.pop('User ID')
data.head()
#Replacing categorical values to numericals
data['Gender'].replace(['Male','Female'],[1,0],inplace=True)


#Using features: Gender, Age for prediction of Purchased label
feature_cols = ['Gender', 'Age']
X = data[feature_cols] # Features
Y = data.Purchased # Target variable

#KNN
seed = 7
max_features = 3
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = KNeighborsClassifier()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("Accuracy for KNeighborsClassifier:", results.mean()) #80

#DECISION TREE
seed = 7
max_features = 3
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = DecisionTreeClassifier(criterion='gini')
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("Accuracy for DecisionTreeClassifier:", results.mean()) #78

#######################################################
'''
BAGGING
'''
######################################################

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("Accuracy for BaggingClassifier:", results.mean()) #80%


from sklearn.ensemble import RandomForestClassifier

seed = 7
num_trees = 100
max_features = 3
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("Accuracy for RandomForestClassifier:", results.mean()) #80


from sklearn.ensemble import ExtraTreesClassifier

seed = 7
num_trees = 100
max_features = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = ExtraTreesClassifier(n_estimators=num_trees)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("Accuracy for ExtraTreesClassifier:", results.mean()) #78.49

#######################################################
'''
BOOSTING
'''
######################################################

from sklearn.ensemble import AdaBoostClassifier

seed = 7
num_trees = 30
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("Accuracy for AdaBoostClassifier:", results.mean()) #83.5


from sklearn.ensemble import GradientBoostingClassifier

seed = 7
num_trees = 100
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("Accuracy for GradientBoostingClassifier:", results.mean()) #81.5


#######################################################
'''
VOTING ENSEMBLE
'''
######################################################

# Voting Ensemble for Classification
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
print("Accuracy for VotingClassifier:", results.mean()) #83.5