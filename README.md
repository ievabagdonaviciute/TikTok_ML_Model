# TikTok Video Virality Prediction

This project uses machine learning techniques to predict the virality of TikTok videos based on various metrics from a dataset obtained from Kaggle. The project explores both **Linear Regression** and **Random Forest Regressor** models to make predictions.

## Project Overview

The aim of this project is to predict how viral a TikTok video will become by analyzing different features from a dataset, such as the number of likes, shares, comments, and other video metrics. Two machine learning models are implemented:
- **Linear Regression**
- **Random Forest Regressor**

The performance of both models is compared to determine which one provides better predictions.

### Key Files
- **`main.py`**: The main script to run the project, including data preprocessing, feature selection, and performance evaluation.
- **`linear_regression.py`**: Implements the linear regression model to predict video virality based on the input metrics.
- **`random_forest_regressor.py`**: Implements a random forest regressor model to predict video virality and compare its performance with the linear regression model.
- **`tiktok_dataset.csv`**: The dataset containing video metrics, sourced from Kaggle.

## Dataset

The dataset used in this project is from Kaggle and contains various metrics related to TikTok videos, such as:
- Number of likes
- Number of shares
- Number of comments
- Video duration, and more.

**Dataset source**: [TikTok Video Dataset](https://www.kaggle.com/datasets/yakhyojon/tiktok)

## Dependencies

To run this project, you'll need the following Python libraries:
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

You can install them using:
```bash
pip install pandas numpy scikit-learn matplotlib
```

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/ievabagdonaviciute/Handwritten-Digit-Recognizer.git
   ```

2. Navigate to the project folder:
   ```bash
   cd Handwritten-Digit-Recognizer
   ```

3. Run the **main.py** script to start the data preprocessing and model evaluation:
   ```bash
   python main.py
   ```

4. To run each model separately, you can execute:
   - Linear Regression:
     ```bash
     python linear_regression.py
     ```
   - Random Forest Regressor:
     ```bash
     python random_forest_regressor.py
     ```

## Results

After training and testing both models, the Random Forest Regressor provides a more accurate prediction of video virality based on the available metrics. Further improvements can be made by adding more features or optimizing the models.

## Credits

The dataset used in this project was sourced from Kaggle: [TikTok Video Dataset](https://www.kaggle.com/datasets/yakhyojon/tiktok).

