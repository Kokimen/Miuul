####################################################
########### DEAL WITH UNBALANCED DATASET ###########
####################################################

######## 1. IMPORT LIBRARY ########

import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc, rcParams
import itertools
import pandas as pd

import warnings

warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)
warnings.filterwarnings("ignore", category = UserWarning)

######## 2. DATASET ########
df = pd.read_csv("datasets/kaggle/creditcard.csv")
df["Class"].value_counts()

######## 3.STANDARDIZATION ########
rob_scaler = RobustScaler()
df['Amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))

####### 4.HOLDOUT SPLIT TRAIN/TEST #########
X = df.drop("Class", axis = 1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123456)

####### MODEL ########
model = LogisticRegression(random_state = 123456)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.3f" % (accuracy))


# .99 --> Too perfect, there is something wrong.


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.rcParams.update({'font.size': 19})
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title, fontdict = {'size': '16'})
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45, fontsize = 12, color = "blue")
    plt.yticks(tick_marks, classes, fontsize = 12, color = "blue")
    rc('font', weight = 'bold')
    fmt = '.1f'
    thresh = cm.max()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = "center",
                 color = "red")

    plt.ylabel('True label', fontdict = {'size': '16'})
    plt.xlabel('Predicted label', fontdict = {'size': '16'})
    plt.tight_layout()
    plt.show()


plot_confusion_matrix(confusion_matrix(y_test, y_pred = y_pred), classes = ['Non Fraud', 'Fraud'],
                      title = 'Confusion matrix')

print(classification_report(y_test, y_pred))


def generate_auc_roc_curve(clf, X_test):
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr)
    plt.show()


generate_auc_roc_curve(model, X_test)

y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print("AUC ROC Curve with Area Under the curve = %.3f" % auc)

##########5.RANDOM SAMPLER############
# !pip install imbalanced-learn
from imblearn.over_sampling import RandomOverSampler

oversample = RandomOverSampler(sampling_strategy = 'minority')
X_randomover, y_randomover = oversample.fit_resample(X_train, y_train)

model.fit(X_randomover, y_randomover)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.3f%%" % (accuracy))

plot_confusion_matrix(confusion_matrix(y_test, y_pred = y_pred), classes = ['Non Fraud', 'Fraud'],
                      title = 'Confusion matrix')

print(classification_report(y_test, y_pred))

#########6.SMOTE OVERSAMPLING#########
from imblearn.over_sampling import SMOTE

oversample = SMOTE()
X_smote, y_smote = oversample.fit_resample(X_train, y_train)

model.fit(X_smote, y_smote)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.3f%%" % (accuracy))

plot_confusion_matrix(confusion_matrix(y_test, y_pred = y_pred), classes = ['Non Fraud', 'Fraud'],
                      title = 'Confusion matrix')

print(classification_report(y_test, y_pred))

#########7.RANDOM UNDERSAMPLING#########
from imblearn.under_sampling import RandomUnderSampler

ranUnSample = RandomUnderSampler()
X_ranUnSample, y_ranUnSample = ranUnSample.fit_resample(X_train, y_train)
