import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

# create a pandas dataframe with 2 categorical and 2 continuous variables
df = pd.DataFrame({
    'Cat_Var1': np.random.choice(['A', 'B'], size=100),
    'Cat_Var2': np.random.choice(['C', 'D'], size=100),
    'Cont_Var1': np.random.normal(0, 1, size=100),
    'Cont_Var2': np.random.normal(10, 2, size=100)
})

# define the number of components for GMM
n_components = 3

# create a GMM model for the categorical variables
cat_model = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)

# fit the GMM model to the categorical data
cat_model.fit(df[['Cat_Var1', 'Cat_Var2']])

# generate synthetic data for the categorical variables
cat_synth = cat_model.sample(n_samples=100, random_state=42)[0]

# round the synthetic categorical data to the nearest integer to obtain binary values
cat_synth = np.round(cat_synth)

# create a GMM model for the continuous variables
cont_model = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)

# fit the GMM model to the continuous data
cont_model.fit(df[['Cont_Var1', 'Cont_Var2']])

# generate synthetic data for the continuous variables
cont_synth = cont_model.sample(n_samples=100, random_state=42)[0]

# combine the synthetic categorical and continuous data
synth_data = pd.DataFrame({
    'Cat_Var1': cat_synth[:, 0].astype(int),
    'Cat_Var2': cat_synth[:, 1].astype(int),
    'Cont_Var1': cont_synth[:, 0],
    'Cont_Var2': cont_synth[:, 1]
})
