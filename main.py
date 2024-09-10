
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from linear_regression import linear_regression_model
from random_forest_regressor import random_forest_model

# Loading the dataset
data = pd.read_csv('tiktok_dataset.csv')

# Data Preprocessing
# Only filling missing values for the numeric columns
numeric_cols = ['video_view_count', 'video_like_count', 'video_share_count', 'video_comment_count', 'video_download_count', 'video_duration_sec']
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Selecting relevant features for analysis (shares, comments, duration, likes, verified status)
X = data[['video_share_count', 'video_comment_count', 'video_duration_sec', 'video_like_count', 'verified_status']]
y = data['video_view_count']  # Target: video view count

# Encoding the 'verified_status' column (Yes/No to 1/0)
X['verified_status'] = X['verified_status'].apply(lambda x: 1 if x == 'verified' else 0)

# Train-testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Calling the linear regression function
print("Running Linear Regression...")
mse, r2 = linear_regression_model(X_train, X_test, y_train, y_test)
print(f"Linear Regression - Mean Squared Error: {mse}, R-squared: {r2}")

# Calling the random forest function
print("Running Random Forest Regressor...")
random_forest_model(X_train, X_test, y_train)
