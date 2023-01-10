import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',10)

file = 'DiamondsPrices.csv'
csv = pd.read_csv(file)
df = pd.DataFrame(csv)

print(df)

# predict diamond cut

# the 4 C's of diamonds = carat (physical weight), cut (shape), color (colorest diamonds are the rarest [DEF-Z]),
# clarity (Diamonds without inclusions (inside) or blemishes (outside) are rare)
# https://www.americangemsociety.org/4cs-of-diamonds/

# carat -> 0 < X < Inf
# cut -> (in order of importance from the highest to the lowest: Ideal, Premium, Very Good, Good, Fair). Ideal is also known as Excellent
# color -> (in order of importance from the highest to the lowest: D, E, F, G, H, I, J)
# clarity -> (in order of importance from the highest to the lowest: IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1)
# depth -> define the cut shape. ideally, a stone cut is around 60% of it diameter
# table -> define the cut shape
# price -> 0 < X < Inf
# x -> 0 < X < Inf
# y -> 0 < X < Inf
# z -> 0 < X < Inf
# x and y are the radius of the diameter, the closer they are, the more the diameter is a perfect circle.
# "depth" x "x" = "z". for example 3.95 x 0.615 = 2.43

# missing values = 0%
print(df.info())

print(df['cut'].value_counts())
print(df['color'].value_counts())
print(df['clarity'].value_counts())

print(df.isna().sum())

df.describe()

temp_cut = pd.DataFrame(df.cut)
diamonds_num = df.drop(['cut', 'color', 'clarity'], axis=1)
diamonds_cat = df[['color', 'clarity']]

temp_cut = temp_cut.replace(['Fair', 'Good', 'Very Good'], 0)
temp_cut = temp_cut.replace(['Ideal', 'Premium'], 1)
# cut is now binary between average and amazing, 0 and 1

# one-hot encoding
one_hot_diamonds = pd.get_dummies(diamonds_cat)
one_hot_diamonds = one_hot_diamonds.join(temp_cut)

dd=pd.DataFrame(diamonds_num)
all_df = dd.join(one_hot_diamonds)

# standardization
scalar = StandardScaler()
X = pd.DataFrame(scalar.fit_transform(diamonds_num), columns=diamonds_num.columns)
X = X.join(one_hot_diamonds)


# PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

scaler = MinMaxScaler()
P = scaler.fit_transform(diamonds_num.join(one_hot_diamonds))
pca = PCA(n_components=0.95)
X_PCA = pca.fit_transform(P)

print("Original Dim", P.shape)
print("Transformed Dim", X_PCA.shape) # from 23 to 13, only need almost half the features

from prettytable import PrettyTable

x = PrettyTable(["Original Dimension", "Transformed Dim"])
x.add_row([P.shape, X_PCA.shape])
print(x.get_string(title = 'PCA Analysis'))

# Random Forest Analysis
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=1, max_depth=10)
model.fit(X.drop("cut", axis=1), X.cut)
features = X.columns
importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 5))
sns.set_style('whitegrid')
plt.title("Feature Importance")
a = plt.barh(range(len(indices)), importances[indices], color='orange', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.grid()
plt.bar_label(a,  label_type='edge')
plt.xlabel("Relative Importance")
plt.show()

# just dropping one_hot columns based off Random Forest Analysis,
# adding back one by one based on OLS

# Correlation Matrix
plt.figure(figsize=(15,15))
corr_matrix = all_df.corr(method='pearson')
plt.title('Correlation Matrix')
sns.heatmap(corr_matrix, annot=True, fmt = ".1f")
plt.show()

x = PrettyTable()
x.add_column('Feature', (corr_matrix['cut']).keys())
x.add_column('Correlation', np.round(corr_matrix['cut'], 3))
print(x.get_string(title = 'Cut Correlation with Other Features'))


# Covariance Matrix
plt.figure(figsize=(30,15))
covMatrix = pd.DataFrame.cov(all_df)
plt.title('Covariance Matrix')
sns.heatmap(covMatrix, annot=True, fmt = ".1f")
plt.show()
print(covMatrix)

# sns.pairplot(all_df, height=2.0)
# plt.show()

Y_fin = X.cut
X_fin = X.drop(['cut', 'color_D', 'color_E','clarity_VS2',
               'clarity_I1'], axis=1)

