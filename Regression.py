import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, Lasso

class RegressionModel:

    def __init__(self, link):
        """
        Initialize the RegressionModel with a dataset from the given link.

        Parameters:
        - link (str): The link to the dataset in CSV format.
        """
        self.df = pd.read_csv(link)

    def exploratory_data_analysis(self):
        """
        Perform comprehensive Exploratory Data Analysis (EDA) on the dataset.
        Display the first few rows, statistical summary, correlation matrix,
        pair plots, and distribution plots.
        """
        print("First few rows:")
        print(self.df.head())

        print("\nStatistical Summary:")
        print(self.df.describe())

        print("\nCorrelation Matrix:")
        correlation_matrix = self.df.corr()
        print(correlation_matrix)

        print("\nPair Plots:")
        sns.pairplot(self.df)
        plt.show()

        print("\nDistribution Plots:")
        self.plot_distributions()
        plt.show()

    def plot_distributions(self):
        """
        Plot distribution plots for each column in the dataset.
        """
        num_columns = len(self.df.columns)
        num_rows = (num_columns + 1) // 2  # Ensure the last row is used if odd number of columns

        fig, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows * 4))
        axes = axes.flatten()

        for i, column in enumerate(self.df.columns):
            sns.histplot(data=self.df, x=column, kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {column}')

        # If the number of columns is odd, remove the empty subplot in the last row
        if num_columns % 2 != 0:
            fig.delaxes(axes[-1])

        plt.tight_layout()

    def principal_component_regression(self, dependent_variable, independent_variables, n_components=None, test_size=0.2, random_state=42):
        """
        Perform Principal Component Regression (PCR).

        Parameters:
        - dependent_variable (str): The column name of the dependent variable.
        - independent_variables (list): List of column names of independent variables.
        - n_components (int or None): Number of components to retain. If None, all components are used.
        - test_size (float): The proportion of the dataset to include in the test split.
        - random_state (int): Seed for random number generation for reproducibility.

        Returns:
        - r2_score_val (float): R-squared score.
        - mean_absolute_error (float): Mean Absolute Error.
        - residual_sum_of_square (float): Residual Sum of Squares.
        """
        train, test = train_test_split(self.df, test_size=test_size, random_state=random_state)

        pca = PCA(n_components=n_components)
        x_train_pca = pca.fit_transform(train[independent_variables])
        x_test_pca = pca.transform(test[independent_variables])

        regr = linear_model.LinearRegression()
        y_train = np.asanyarray(train[[dependent_variable]])
        regr.fit(x_train_pca, y_train)

        y_hat = regr.predict(x_test_pca)
        test_y = np.asanyarray(test[[dependent_variable]])

        r2_score_val = r2_score(test_y, y_hat)
        mean_absolute_error = np.mean(np.absolute(y_hat - test_y))
        residual_sum_of_square = np.mean((y_hat - test_y) ** 2)

        return r2_score_val, mean_absolute_error, residual_sum_of_square

    def partial_least_squares_regression(self, dependent_variable, independent_variables, n_components=None, test_size=0.2, random_state=42):
        """
        Perform Partial Least Squares Regression (PLSR).

        Parameters:
        - dependent_variable (str): The column name of the dependent variable.
        - independent_variables (list): List of column names of independent variables.
        - n_components (int or None): Number of components to retain. If None, all components are used.
        - test_size (float): The proportion of the dataset to include in the test split.
        - random_state (int): Seed for random number generation for reproducibility.

        Returns:
        - r2_score_val (float): R-squared score.
        - mean_absolute_error (float): Mean Absolute Error.
        - residual_sum_of_square (float): Residual Sum of Squares.
        """
        train, test = train_test_split(self.df, test_size=test_size, random_state=random_state)

        plsr = PLSRegression(n_components=n_components)
        x_train_plsr = np.asanyarray(train[independent_variables])
        y_train = np.asanyarray(train[[dependent_variable]])
        plsr.fit(x_train_plsr, y_train)

        x_test_plsr = np.asanyarray(test[independent_variables])
        y_hat = plsr.predict(x_test_plsr)
        test_y = np.asanyarray(test[[dependent_variable]])

        r2_score_val = r2_score(test_y, y_hat)
        mean_absolute_error = np.mean(np.absolute(y_hat - test_y))
        residual_sum_of_square = np.mean((y_hat - test_y) ** 2)

        return r2_score_val, mean_absolute_error, residual_sum_of_square


    def ridge_regression(self, dependent_variable, independent_variables, alpha=1.0, test_size=0.2, random_state=42):
        """
        Perform Ridge Regression.

        Parameters:
        - dependent_variable (str): The column name of the dependent variable.
        - independent_variables (list): List of column names of independent variables.
        - alpha (float): Regularization strength (default is 1.0).
        - test_size (float): The proportion of the dataset to include in the test split.
        - random_state (int): Seed for random number generation for reproducibility.

        Returns:
        - r2_score_val (float): R-squared score.
        - mean_absolute_error (float): Mean Absolute Error.
        - residual_sum_of_square (float): Residual Sum of Squares.
        """
        train, test = train_test_split(self.df, test_size=test_size, random_state=random_state)

        ridge = Ridge(alpha=alpha)
        x = np.asanyarray(train[independent_variables])
        y = np.asanyarray(train[[dependent_variable]])
        ridge.fit(x, y)

        y_hat = ridge.predict(test[independent_variables])
        test_x = np.asanyarray(test[independent_variables])
        test_y = np.asanyarray(test[[dependent_variable]])

        r2_score_val = r2_score(test_y, y_hat)
        mean_absolute_error = np.mean(np.absolute(y_hat - test_y))
        residual_sum_of_square = np.mean((y_hat - test_y) ** 2)

        return r2_score_val, mean_absolute_error, residual_sum_of_square

    def lasso_regression(self, dependent_variable, independent_variables, alpha=1.0, test_size=0.2, random_state=42):
        """
        Perform Lasso Regression.

        Parameters:
        - dependent_variable (str): The column name of the dependent variable.
        - independent_variables (list): List of column names of independent variables.
        - alpha (float): Regularization strength (default is 1.0).
        - test_size (float): The proportion of the dataset to include in the test split.
        - random_state (int): Seed for random number generation for reproducibility.

        Returns:
        - r2_score_val (float): R-squared score.
        - mean_absolute_error (float): Mean Absolute Error.
        - residual_sum_of_square (float): Residual Sum of Squares.
        """
        train, test = train_test_split(self.df, test_size=test_size, random_state=random_state)

        lasso = Lasso(alpha=alpha)
        x = np.asanyarray(train[independent_variables])
        y = np.asanyarray(train[[dependent_variable]])
        lasso.fit(x, y)

        y_hat = lasso.predict(test[independent_variables])
        test_x = np.asanyarray(test[independent_variables])
        test_y = np.asanyarray(test[[dependent_variable]])

        r2_score_val = r2_score(test_y, y_hat)
        mean_absolute_error = np.mean(np.absolute(y_hat - test_y))
        residual_sum_of_square = np.mean((y_hat - test_y) ** 2)

        return r2_score_val, mean_absolute_error, residual_sum_of_square

    def simple_linear_regression(self, dependent_variable, independent_variable, test_size=0.2, random_state=42):
        """
        Perform Simple Linear Regression.

        Parameters:
        - dependent_variable (str): The column name of the dependent variable.
        - independent_variable (str): The column name of the independent variable.
        - test_size (float): The proportion of the dataset to include in the test split.
        - random_state (int): Seed for random number generation for reproducibility.

        Returns:
        - r2_score_val (float): R-squared score.
        - mean_absolute_error (float): Mean Absolute Error.
        - residual_sum_of_square (float): Residual Sum of Squares.
        """
        train, test = train_test_split(self.df, test_size=test_size, random_state=random_state)

        regr = linear_model.LinearRegression()
        train_x = np.asanyarray(train[[independent_variable]])
        train_y = np.asanyarray(train[[dependent_variable]])
        regr.fit(train_x, train_y)

        test_x = np.asanyarray(test[[independent_variable]])
        test_y = np.asanyarray(test[[dependent_variable]])
        test_y_ = regr.predict(test_x)

        r2_score_val = r2_score(test_y, test_y_)
        mean_absolute_error = np.mean(np.absolute(test_y_ - test_y))
        residual_sum_of_square = np.mean((test_y_ - test_y) ** 2)

        return r2_score_val, mean_absolute_error, residual_sum_of_square

    def multiple_regression(self, dependent_variables, independent_variable, test_size=0.2, random_state=42):
        """
        Perform Multiple Linear Regression.

        Parameters:
        - dependent_variables (list): List of column names of dependent variables.
        - independent_variable (str): The column name of the independent variable.
        - test_size (float): The proportion of the dataset to include in the test split.
        - random_state (int): Seed for random number generation for reproducibility.

        Returns:
        - residual_sum_of_squares (float): Residual Sum of Squares.
        - variance_score (float): Variance Score.
        """
        train, test = train_test_split(self.df, test_size=test_size, random_state=random_state)

        regr = linear_model.LinearRegression()
        x = np.asanyarray(train[dependent_variables])
        y = np.asanyarray(train[[independent_variable]])
        regr.fit(x, y)

        y_hat = regr.predict(test[dependent_variables])
        test_x = np.asanyarray(test[dependent_variables])
        test_y = np.asanyarray(test[[independent_variable]])

        residual_sum_of_squares = np.mean((y_hat - test_y) ** 2)
        variance_score = regr.score(x, y)

        return residual_sum_of_squares, variance_score

    def polynomial_regression(self, dependent_variables, independent_variable, test_size=0.2, random_state=42, degree=2):
        """
        Perform Polynomial Regression.

        Parameters:
        - dependent_variables (list): List of column names of dependent variables.
        - independent_variable (str): The column name of the independent variable.
        - test_size (float): The proportion of the dataset to include in the test split.
        - random_state (int): Seed for random number generation for reproducibility.
        - degree (int): Degree of the polynomial features.

        Returns:
        - r2_score_val (float): R-squared score.
        - mean_absolute_error (float): Mean Absolute Error.
        - residual_sum_of_square (float): Residual Sum of Squares.
        """
        train, test = train_test_split(self.df, test_size=test_size, random_state=random_state)

        train_x = np.asanyarray(train[['independent_variable']])
        train_y = np.asanyarray(train[['dependent_variable']])

        test_x = np.asanyarray(test[['independent_variable']])
        test_y = np.asanyarray(test[['dependent_variable']])

        poly = PolynomialFeatures(degree=degree)
        train_x_poly = poly.fit_transform(train_x)

        clf = linear_model.LinearRegression()
        train_y_ = clf.fit(train_x_poly, train_y)

        test_x_poly = poly.transform(test_x)
        test_y_ = clf.predict(test_x_poly)

        r2_score_val = r2_score(test_y, test_y_)
        mean_absolute_error = np.mean(np.absolute(test_y_ - test_y))
        residual_sum_of_square = np.mean((test_y_ - test_y) ** 2)

        return r2_score_val, mean_absolute_error, residual_sum_of_square
