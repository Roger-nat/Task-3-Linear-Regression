import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/housing.csv")

print(df.head())
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

X = df.drop("medv", axis=1)
y = df["medv"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42


)
print("\nTraining data:",X_train.shape)
print("Testing data:",X_test.shape)

model=LinearRegression()
model.fit(X_train, y_train)

y_pred=model.predict(X_test)

mae=mean_absolute_error(y_test, y_pred)
print("\nMean Absoulte Error:", mae)

mse=mean_squared_error(y_test, y_pred)
print("Mean Sqaaured Error:", mse)

r2=r2_score(y_test, y_pred)
print("R2 Score:", r2)

print("\nModel Coefficients:")
for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", coefficient)

print("Intercept:", model.intercept_)

X_simple=df[['rm']]
y_simple=df['medv']

X_train_simple, X_test_simple, y_train_simple, y_test_simple=train_test_split(
    X_simple,
    y_simple,
    test_size=0.2,
    random_state=42
)
simple_model=LinearRegression()
simple_model.fit(X_train_simple, y_train_simple)
y_pred_simple=simple_model.predict(X_test_simple)

plt.scatter(X_test_simple, y_test_simple)
plt.plot(X_test_simple, y_pred_simple)
plt.xlabel("Average Number of rows")
plt.ylabel("house price")
plt.title("simple Linear Regression")
plt.show()

mae_simple = mean_absolute_error(y_test_simple, y_pred_simple)
mse_simple = mean_squared_error(y_test_simple, y_pred_simple)
r2_simple = r2_score(y_test_simple, y_pred_simple)

print("\nSimple Linear Regression Results:")
print("MAE:", mae_simple)
print("MSE:", mse_simple)
print("R2 Score:", r2_simple)

print("\nModel Comparison:")
print("Simple Regression R2:", r2_simple)
print("Multiple Regression R2:", r2)