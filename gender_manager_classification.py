"""
Create a sample data frame in python with 1000 observations.

50% are men and 50% are women
15% are managers: 20% of the men are managers, 10% of women are managers
50% have higher edu: 45% of not managers, 70% of female managers, 60% of male managers
Age is 45 (min = 20, max = 65): not managers = 45, female managers = 55, male managers = 50
High age is age over 55 (approximately 75th percentile)
Ability is 5 (min = 1, max = 10): not managers = 5, managers = 7
High ability is ability over 7 (approximately 75th percentile)
"""

# Basic
import numpy as np
import pandas as pd

# Graphing
import seaborn as sns
import matplotlib.pyplot as plt

# Algorithms
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from scipy.stats import truncnorm

# Functions
# function to generate random numbers with min, max, mean, and std
def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

# function to annotate bars
def annotate_bars(ax=None, fmt='.2f', **kwargs):
    ax = plt.gca() if ax is None else ax
    for p in ax.patches:
         ax.annotate('{{:{:s}}}'.format(fmt).format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                     xytext=(0, 5),textcoords='offset points',
                     ha='center', va='center', **kwargs)
        
# Set seed for reproducibility
np.random.seed(1234)

# Define number of observations
n = 1000

# Create male variable
male = np.random.choice([1, 0], size=n, p=[0.5, 0.5])

# Create manager by male variable
manager = np.where(male == 1, np.random.choice([1, 0], size=n, p=[0.2, 0.8]), 
                              np.random.choice([1, 0], size=n, p=[0.1, 0.9]))

# Create education level variable
edu_high =  np.where(manager & (male == 0), np.random.choice([1, 0], size=n, p=[0.7, 0.3]),
            np.where(manager & (male == 1), np.random.choice([1, 0], size=n, p=[0.6, 0.4]),
            np.where(~manager & (male == 0),np.random.choice([1, 0], size=n, p=[0.45, 0.55]),
                                            np.random.choice([1, 0], size=n, p=[0.45, 0.55])
            )))

# Age - this is a proxy for experience
age =   np.where(manager & (male == 0), get_truncated_normal(mean=55, sd=10, low=40, upp=65).rvs(n).astype(int),
        np.where(manager & (male == 1), get_truncated_normal(mean=50, sd=10, low=40, upp=65).rvs(n).astype(int),
        np.where(~manager & (male == 0),get_truncated_normal(mean=45, sd=10, low=20, upp=65).rvs(n).astype(int),
                                        get_truncated_normal(mean=45, sd=10, low=20, upp=65).rvs(n).astype(int)
            )))

# Test
X = get_truncated_normal(mean=7, sd=4, low=1, upp=11).rvs(n).astype(int)
pd.DataFrame(X).describe()

# We use ability, because wages are too highly correlated with managers
ability = np.where(manager==0, get_truncated_normal(mean=5, sd = 4, low=1, upp=11).rvs(n).astype(int), 
                               get_truncated_normal(mean=7, sd = 4, low=1, upp=11).rvs(n).astype(int)
                )

# Create data frame
df = pd.DataFrame({
    'male': male,
    'age': age,
    'manager': manager,
    'edu_high': edu_high,
    'ability': ability
})

# Create categorical variables for age and ability
df["age_high"] = np.where(df["age"]>50,1,0)
df["ability_high"] = np.where(df["ability"]>6,1,0)


"""
# Describing the data
"""
# Print first few rows of data frame
print(df.head())
df.describe()

df_descriptives = df.groupby(['manager']).mean()
df_descriptives

df_descriptives = df.groupby(['male',"manager"]).mean()
df_descriptives

plt.figure(figsize=(15,10)) #adjust the size of plot
sns.countplot(data=df, x="age", hue="manager")
plt.tight_layout()
plt.show()

sns.countplot(data=df, x="ability", hue="manager")
plt.show()

"""
# Graphing the data
"""

# group by manager status

grouped = df.groupby(['manager']).mean().reset_index()
grouped_long = pd.melt(grouped, id_vars=['manager']).reset_index(drop=True)
g = sns.FacetGrid(grouped_long, col="variable", sharey=False)
g.map_dataframe(sns.barplot, x="manager",y="value",errorbar=None)
g.map(annotate_bars, fmt='.2g', fontsize=8, color='k')
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.fig.show()

# group by gender and manager status

grouped = df.groupby(['male', 'manager']).mean().reset_index()
grouped_long = pd.melt(grouped, id_vars=['male', 'manager']).reset_index(drop=True)
g = sns.FacetGrid(grouped_long, col="variable", sharey=False)
g.map_dataframe(sns.barplot, x="manager",y="value",hue="male", errorbar=None).add_legend(title="Male")
g.map(annotate_bars, fmt='.2g', fontsize=8, color='k')
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.fig.show()

df_corr = df.drop(["age_high","ability_high"], axis=1)

# correlation
plt.clf()
sns.heatmap(df_corr.astype(float).corr(), annot=True)
plt.show()

"""
# Fitting model with both categorical variables
"""
df
X = df[["male","edu_high","age_high","ability_high"]]
X
y = df["manager"]

models = []
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVC', SVC()))

results = []
# evaluate each model in turn
for name, model in models:
    # fit the model
    classifier = model
    classifier.fit(X, y)
    
    # predict 
    y_predict = classifier.predict(X)
    df[f"{'yhat'}_{name}"] = y_predict
    
    # estimate model fit
    tn, fp, fn, tp = confusion_matrix(y, y_predict).ravel()

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    accuracy = accuracy_score(y, y_predict) 
    
    results.append([name, accuracy, tnr, tpr, fnr, fpr])

results = pd.DataFrame(results,columns=("Model", "Accuracy", "True neg rate", "True pos rate", "False neg rate", "False pos rate"))
results_long = pd.melt(results, id_vars=['Model']).reset_index(drop=True)
results_long

g = sns.FacetGrid(results_long, col="variable", sharey=False, height=6)
g.map_dataframe(sns.barplot, x="Model",y="value",errorbar=None)
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.map(annotate_bars, fmt='.2g', fontsize=12, color='k')
g.fig.tight_layout()
g.fig.subplots_adjust(wspace=.25)
g.fig.show()

"""
# Fitting model with both categorical and continuous variables
"""
df
X = df[["male","edu_high","age","ability"]]
X
y = df["manager"]

models = []
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVC', SVC()))

results = []
# evaluate each model in turn
for name, model in models:
    # fit the model
    classifier = model
    classifier.fit(X, y)
    
    # predict 
    y_predict = classifier.predict(X)
    df[f"{'yhat'}_{name}"] = y_predict
    
    # estimate model fit
    tn, fp, fn, tp = confusion_matrix(y, y_predict).ravel()

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    accuracy = accuracy_score(y, y_predict) 
    
    results.append([name, accuracy, tnr, tpr, fnr, fpr])

results = pd.DataFrame(results,columns=("Model", "Accuracy", "True neg rate", "True pos rate", "False neg rate", "False pos rate"))
results_long = pd.melt(results, id_vars=['Model']).reset_index(drop=True)
results_long

g = sns.FacetGrid(results_long, col="variable", sharey=False, height=6)
g.map_dataframe(sns.barplot, x="Model",y="value",errorbar=None)
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.map(annotate_bars, fmt='.2g', fontsize=12, color='k')
g.fig.tight_layout()
g.fig.subplots_adjust(wspace=.25)
g.fig.show()
