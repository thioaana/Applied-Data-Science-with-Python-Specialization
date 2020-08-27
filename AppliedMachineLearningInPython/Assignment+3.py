import numpy as np
import pandas as pd
import sys

from sklearn.model_selection import train_test_split
df = pd.read_csv('data/fraud_data.csv')
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


def answer_one():
    df = pd.read_csv('data/fraud_data.csv')
    class1 = df.Class[df.Class == 1].count()
    class0 = df.Class[df.Class == 0].count()
    return class1 / (class1 + class0)


def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score

    clfDummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    yPred = clfDummy.predict(X_test)
    return (clfDummy.score(X_test, y_test), recall_score(y_test, yPred))


def answer_three():
    from sklearn.metrics import recall_score, precision_score, accuracy_score
    from sklearn.svm import SVC

    clf = SVC()
    clf.fit(X_train, y_train)
    yPred = clf.predict(X_test)
    return (accuracy_score(y_test, yPred), recall_score(y_test, yPred), precision_score(y_test, yPred))



def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC
    clf = SVC(C=1e9, gamma=1e-07)  # myParams)
    clf.fit(X_train, y_train)
    yDecisions = clf.decision_function(X_test)
    yPred = yDecisions > -220
    yPred = yPred.astype(int)
    myConfMat = confusion_matrix(y_test, yPred)
    return myConfMat


# For the logisitic regression classifier, create a precision recall curve and a roc curve using y_test and the probability estimates for X_test (probability it is fraud).
# Looking at the precision recall curve, what is the recall when the precision is `0.75`?
#  Looking at the roc curve, what is the true positive rate when the false positive rate is `0.16`?
# *This function should return a tuple with two floats, i.e. `(recall, true positive rate)`.*

def answer_five():
    from sklearn.metrics import precision_recall_curve, roc_curve
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=10000).fit(X_train, y_train)
    yDecision = lr.decision_function(X_test)

    precision, recall, thresholds = precision_recall_curve(y_test, yDecision)
    ll = list(zip(precision, recall))
    ll.sort()
    recallValue = [i[1] for i in ll if i[0] == 0.75][0]

    fpr, tpr, _ = roc_curve(y_test, yDecision)
    ft = list(zip(fpr, tpr))
    ft.sort()
    # for i in ft: print(i)

    return  (recallValue, 0.9375)


# Perform a grid search over the parameters listed below for a Logisitic Regression classifier,
# using recall for scoring and the default 3-fold cross validation.
# `'penalty': ['l1', 'l2']`
# `'C':[0.01, 0.1, 1, 10, 100]`
# From `.cv_results_`, create an array of the mean test scores of each parameter combination. i.e.
# 
# |      	| `l1` 	| `l2` 	|
# |:----:	|----	|----	|
# | **`0.01`** 	|    ?	|   ? 	|
# | **`0.1`**  	|    ?	|   ? 	|
# | **`1`**    	|    ?	|   ? 	|
# | **`10`**   	|    ?	|   ? 	|
# | **`100`**   	|    ?	|   ? 	|
#
# *This function should return a 5 by 2 numpy array with 10 floats.* 
# *Note: do not return a DataFrame, just the values denoted by '?' above in a numpy array. You might need to reshape your raw result to meet the format we are looking for.*


def answer_six():
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=10000).fit(X_train, y_train)
    # Your code here
    
    return # Return your answer


# In[ ]:

# Use the following function to help visualize results from the grid search
def GridSearch_Heatmap(scores):
    get_ipython().magic('matplotlib notebook')
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure()
    sns.heatmap(scores.reshape(5,2), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10, 100])
    plt.yticks(rotation=0);

#GridSearch_Heatmap(answer_six())
# print(answer_one())
# print(answer_two())
# print(answer_three())
# print(answer_four())
# print(answer_five())
print(answer_six())