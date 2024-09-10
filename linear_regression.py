
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def linear_regression_model(X_train, X_test, y_train, y_test):

    # Initializing and fit the Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Predicting the view count with the test data
    y_pred = lr_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Plotting Actual vs Predicted View Count
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual View Count')
    plt.ylabel('Predicted View Count')
    plt.title('Actual vs Predicted Video View Count (Linear Regression)')
    plt.show()

    return mse, r2
