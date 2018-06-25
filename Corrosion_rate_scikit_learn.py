import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# Read the datasets
df = pd.read_csv("Datasets_acestic_acid.csv")
# Set an x and y
X = df[["Time (min)"]].values
y = df["Wall – Thickness Loss (µm)"].values
# Call Linear regression
regr = LinearRegression()

# Call an PolynomialFeatures
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

# Prepare x data for predict
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

# Fit the date into regression model
regr = regr.fit(X, y)
# Predict on new data
y_lin_fit = regr.predict(X_fit)
# Calculate the MSE
linear_MSE = mean_squared_error(y, regr.predict(X))
# Calculate R^2
linear_r2 = r2_score(y, regr.predict(X))
# Print coefficient and intercept
print('Slope linear' ,regr.coef_)
print('Intercept linear:' ,regr.intercept_)
# Predict for whole year
print(regr.predict(525600))
print('\n')

# Fit the date into regression model
regr = regr.fit(X_quad, y)
# Predict on new data
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
# Calculate the MSE
quadratic_MSE = mean_squared_error(y, regr.predict(X_quad))
# Calculate R^2
quadratic_r2 = r2_score(y, regr.predict(X_quad))
# Print coefficient and intercept
print('Slope quadratic:',regr.coef_)
print('Intercept quadratic:',regr.intercept_)
print('\n')

# Fit the date into regression mode
regr = regr.fit(X_cubic, y)
# Predict on new data
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
# Calculate the MSE
cubic_MSE= mean_squared_error(y, regr.predict(X_cubic))
# Calculate R^2
cubic_r2 = r2_score(y, regr.predict(X_cubic))
# Print coefficient and intercept
print('Slope cubic:', regr.coef_)
print('Intercept cubic:',regr.intercept_)
print('\n')

# Print MSE for all models
print('Training MSE linear: %.6f, quadratic: %.6f, cubic: %.6f' % (
        linear_MSE,
        quadratic_MSE,
        cubic_MSE))
# Print R^2 for all models
print('Training R^2 linear: %.6f, quadratic: %.6f, cubic: %.6f' % (
        linear_r2,
        quadratic_r2,
        cubic_r2))

# plot results
# set a plot size
plt.figure(figsize=(16,8))
# plot datasets points
plt.scatter(X, y, label='training points', color='orange',s=5)
# plot simple linear regression
plt.plot(X_fit, y_lin_fit,
         label='linear (n=1), $R^2=%.5f$ $MSE=%.5f$' % (linear_r2, linear_MSE),
         color='black',
         lw=2,
         linestyle=':')
# plot quadratic linear regression
plt.plot(X_fit, y_quad_fit,
         label='quadratic (n=2), $R^2=%.5f$  $MSE=%.5f$' % (quadratic_r2, quadratic_MSE),
         color='red',
         lw=2,
         linestyle='-')
# plot cubic linear egression
plt.plot(X_fit, y_cubic_fit,
         label='cubic (n=3), $R^2=%.5f$  $MSE=%.5f$' % (cubic_r2, cubic_MSE),
         color='green',
         lw=2,
         linestyle='--')
# add title to plot
plt.title('Scikit-Learn model')
# set x and y labels
plt.xlabel('TIME [min]')
plt.ylabel('WALL THICKNESS [µm]')
# Add legend
plt.legend(loc='upper right')
# save plot
plt.savefig('cubic_new.png',bbox_inches='tight',dpi=600)
plt.show()
