
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

def random_forest_model(X_train, X_test, y_train):
    
    # Initializing and fit the Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Getting feature importances
    importances = rf_model.feature_importances_
    features = X_train.columns

    # Plotting feature importance percentages
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=features)
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.title('Feature Importance in Predicting Video Virality')
    plt.show()
