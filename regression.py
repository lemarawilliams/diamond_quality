import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from numpy import linalg as LA
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

import eda

X = eda.X_fin
Y = eda.Y_fin

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1, shuffle=True)

# T-test Analysis/ Step-wise Regression

model = sm.OLS(y_train, X_train).fit()
print(model.summary())
# Wow both are 0.47, all 0 with top 13
# Added  'clarity_VS1', goes to 0.539
# Added 'color_G', goes to 0.565
# Added 'clarity_IF', goes to 0.576
# Added 'color_F', goes 0.596
# Added  'color_J', goes to 0.611
# Added 'clarity_VS2', goes to 0.704, but t-scores go up from 0
#     carat-0.072, price-0.346, color_F-0.001, color_J-0.156
#       gonna stop with color_J


# CI pt2
from statsmodels.stats.proportion import proportion_confint

ci = list(proportion_confint(count=35342,    # Number of 1's
                   nobs=53940,    # Total
                   alpha=(1 - 0.95)))
# Alpha, which is 1 minus the confidence level
ci.append(eda.one_hot_diamonds.cut.mean())

from prettytable import PrettyTable

x = PrettyTable(['Low CI', 'High CI', 'Mean'])
x.add_row([ci[0], ci[1], ci[2]])

print(x.get_string(title = 'Confidence Interval Analysis'))


# Collinearity analysis
vif = pd.DataFrame()
vif["features"] = X[[ 'depth', 'table', 'price',  'y']].columns
vif["vif_Factor"] = [variance_inflation_factor(X[[ 'depth', 'table', 'price',  'y']].values, i)
                     for i in range(X[[ 'depth', 'table', 'price',  'y' ]].shape[1])]
x = PrettyTable([ 'features',  'vif_Factor'])
for i in (range(len(vif))):
    x.add_row(vif.loc[i])
print(x.get_string(title = 'Collinearity Analysis with vif'))
print(vif)


# Tells me to cut carat, x and, z, but I can't really trust it
# Not very helpful as most of my features are one-hot encoded so they cannot
# be anaylzed for overfitting through this method


predictions = model.predict(X_test)
print(predictions)

MSE = np.round(np.square(np.subtract(y_test, predictions)).mean(), 3)
print(f"MSE using OLS is {MSE}")


