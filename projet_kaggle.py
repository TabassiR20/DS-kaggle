from sklearn import metrics

from sklearn.metrics import  precision_score, recall_score, f1_score

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix


import pandas as pd

test = pd.read_csv('Test.csv')

train = pd.read_csv('Train.csv')

train.shape

test.shape

train.sample(5)

missing = train.isna().sum()
print(f'  missing datas are:           { ", ".join(missing.index.values) }')

train.duplicated().sum()

train.Month.unique()

train.VisitorType.unique()

def get_dummies(df,test = False):
    df.Month = df['Month'].map({'Feb' : 2, 'Mar' : 3, 'May' : 5, 'Oct': 10, 'June' : 6, 'Jul' : 7, 'Aug' : 8, 'Nov' : 11, 'Sep' : 9,'Dec' : 12}).astype(int)
    df.VisitorType = df['VisitorType'].map({'Returning_Visitor' : 2, 'New_Visitor' : 1, 'Other' : 3}).astype(int)
    df.Weekend = df['Weekend'].map( {True: 1, False: 0} ).astype(int)
    if test == False:
        df.Revenue = df['Revenue'].map( {True: 1, False: 0} ).astype(int)

get_dummies(train)
train.head()

def ReplacingTest(df,test = False):
    df.Month = df['Month'].map({
                                'Nov' : 11,
                                'Dec' : 12}).astype(int)
    df.VisitorType = df['VisitorType'].map({'Returning_Visitor' : 2, 'New_Visitor' : 1, 'Other' : 3}).astype(int)
    df.Weekend = df['Weekend'].map( {True: 1, False: 0} ).astype(int)


ReplacingTest(test)

train = train.drop(['id'], axis=1)

test = test.dropna()

train = train.dropna()

from sklearn.impute import SimpleImputer 
#from sklearn.experimental import enable_iterative_imputer

#from sklearn.impute import IterativeImputer

#imp = IterativeImputer(random_state=0)
#train = imp.fit_transform(train)
#train = pd.DataFrame(train, columns = ['Administrative','Administrative_Duration','Informational','Informational_Duration','ProductRelated','ProductRelated_Duration','BounceRates','ExitRates','PageValues','SpecialDay','Month','OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend','Revenue'])
#train [train.isna().any(axis=1)]

train.max() #displays the max of every column

#NORMALIZATION

#Administrative_Duration , Informational_Duration , ProductRelated , ProductRelated_Duration , PageValues
train['Administrative_Duration'] = train['Administrative_Duration']/27.0
train['PageValues'] = train ['PageValues']/360.953384
train['ProductRelated'] = train ['ProductRelated']/705.0
train['Informational_Duration'] = train ['Informational_Duration']/2549.375000
train['ProductRelated_Duration'] = train ['ProductRelated_Duration']/63973.522230

test['Administrative_Duration'] = test['Administrative_Duration']/27.0
test ['PageValues'] = test ['PageValues']/360.953384
test['ProductRelated'] = test ['ProductRelated']/705.0
test['Informational_Duration'] = test ['Informational_Duration']/2549.375000
test['ProductRelated_Duration'] = test ['ProductRelated_Duration']/63973.522230
test.head()

""" 80% train 20% test splitting"""

X = train.drop(['Revenue'], axis=1)
y = train[['Revenue']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20 , random_state=0)


knn = KNeighborsClassifier()
tree = DecisionTreeClassifier()
svm = SVC()
naive = GaussianNB()
rfc = RandomForestClassifier()

knn.fit(X_train, y_train)
tree.fit(X_train, y_train)
naive.fit(X_train, y_train)
svm.fit(X_train, y_train)
rfc.fit(X_train, y_train)
from sklearn.model_selection import KFold

k_fold = KFold(n_splits= 10, shuffle = True, random_state=0)

from warnings import simplefilter

for k in range(3,15):
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(X_train, y_train)
  y_preds = knn.predict(X_test)
  print(f'{k}: {f1_score(y_preds, y_test, average="weighted")}')
#ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)



(np.mean(score))

from sklearn.metrics import f1_score 

for model, name in [ [knn,'knn'], [tree,'tree'], [naive,'naive'], [svm,'svm'], [gbc, 'gbc'], [rfc,'rfc'] ]:
  y_preds = model.predict(X_test)
  print(
      f'{name}: {f1_score(y_preds, y_test, average="weighted")}'
      )

"""**GridSearchCV**"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier

gbc = GradientBoostingClassifier()
rfc = RandomForestClassifier()

gbc.fit(X_train, y_train)
rfc.fit(X_train, y_train)

from sklearn.model_selection import GridSearchCV

rfc = RandomForestClassifier()

param_grid = {
    "n_estimators": range(100,1000,50), 
    'criterion': ['gini', 'entropy']
}

grid = GridSearchCV(rfc, param_grid, scoring="accuracy", cv=2, n_jobs=-1)

param_grid = {"max_depth": [3, None],"max_features": [1, 3, 10],"min_samples_split": [2, 3, 10],"bootstrap": [True, False],"criterion": ["gini", "entropy"]}
rforest = RandomForestClassifier()
forest_cv = GridSearchCV(rforest,param_grid,cv=10)
forest_cv = forest_cv.fit(X,y)
forest_cv.best_params_

grid.best_estimator_

"""#Submission"""

test = pd.read_csv('Test.csv')

test = test.drop(['id'], axis=1)

get_dummies(test,test= True)
data.head()

"""Normalize Test"""

test2 = pd.read_csv('Test.csv')

rfc.fit(X,y)
rfc.predict(test)

def ReplacingTest(df,test = False):
    df.Month = df['Month'].map({
                                'Nov' : 11,
                                'Dec' : 12}).astype(int)
    df.VisitorType = df['VisitorType'].map({'Returning_Visitor' : 2, 'New_Visitor' : 1, 'Other' : 3}).astype(int)
    df.Weekend = df['Weekend'].map( {True: 1, False: 0} ).astype(int)


ReplacingTest(test)

op=pd.DataFrame(test2 ['id'])
op['Revenue']=pred

op.to_csv("op_rf.csv", index=False)