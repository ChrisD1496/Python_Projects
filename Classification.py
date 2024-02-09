import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.exceptions import DataConversionWarning, UndefinedMetricWarning, ConvergenceWarning
import pdpipe as pdp
import pickle
import warnings

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


class Classification:
    def __init__(self, link):
        """
        Initialize the Classification with a dataset from the given link.

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

    @staticmethod
    def _grid_search(model, param_grid, features_train, target_train, model_name):
        """
        Perform grid search for hyperparameter tuning.

        Parameters:
        - model: The machine learning model.
        - param_grid (dict): Hyperparameter grid.
        - features_train: Training features.
        - target_train: Training target.
        - model_name (str): Name of the model for saving.

        Returns:
        - evaluation_metrics (dict): Dictionary of evaluation metrics.
        """
        pipeline = Pipeline([('std', StandardScaler()), (model_name, model)])
        grid_search_model = GridSearchCV(estimator=pipeline,
                                         param_grid=param_grid,
                                         scoring='f1',
                                         cv=5
                                         )
        grid_search_model.fit(features_train, target_train)
        target_test_pred = grid_search_model.predict(features_test)
        precision = precision_score(target_test, target_test_pred)
        recall = recall_score(target_test, target_test_pred)
        f1 = f1_score(target_test, target_test_pred)
        accuracy = accuracy_score(target_test, target_test_pred)

        evaluation_metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }

        pickle.dump(grid_search_model, open(f'{model_name}.p', 'wb'))

        return evaluation_metrics

    def decision_tree(self, features_train, target_train, features_test, target_test, search_space=None):
        model = DecisionTreeClassifier(random_state=0, class_weight='balanced')
        if search_space is None:
            search_space = {'tree__max_depth': range(1, 10, 10)}
        return self._grid_search(model, search_space, features_train, target_train, 'tree')

    def knn(self, features_train, target_train, features_test, target_test, search_space=None):
        pipeline_kn = Pipeline([('std', StandardScaler()), ('kn', KNeighborsClassifier(weights='distance'))])
        if search_space is None:
            k = np.unique(np.geomspace(1, 30, 5, dtype='int'))
            search_space = {'kn__n_neighbors': k}
        return self._grid_search(pipeline_kn, search_space, features_train, target_train, 'knn')

    def logistic_regression(self, features_train, target_train, features_test, target_test, search_space=None):
        model_lr = LogisticRegression(class_weight='balanced', random_state=0)
        if search_space is None:
            search_space = {'lr__C': [0.001, 0.01, 0.1, 1, 10, 100]}
        return self._grid_search(model_lr, search_space, features_train, target_train, 'logistic_regression')

    def random_forest(self, features_train, target_train, features_test, target_test, search_space=None):
        model_rf = RandomForestClassifier(class_weight='balanced', random_state=0)
        if search_space is None:
            search_space = {'rf__n_estimators': [50, 100, 150],
                            'rf__max_depth': [None, 10, 20, 30]}
        return self._grid_search(model_rf, search_space, features_train, target_train, 'random_forest')

    def support_vector_machine(self, features_train, target_train, features_test, target_test, search_space=None):
        model_svm = SVC(class_weight='balanced', random_state=0)
        if search_space is None:
            search_space = {'svm__C': [0.1, 1, 10],
                            'svm__kernel': ['linear', 'rbf']}
        return self._grid_search(model_svm, search_space, features_train, target_train, 'support_vector_machine')

    def neural_network(self, features_train, target_train, features_test, target_test, param_grid=None):
        def create_model():
            model = Sequential()
            model.add(Dense(units=64, activation='relu', input_dim=features_train.shape[1]))
            model.add(Dense(units=1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model

        model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=0)
        if param_grid is None:
            param_grid = {'epochs': [10, 20, 30],
                          'batch_size': [32, 64, 128]}
        return self._grid_search(model, param_grid, features_train, target_train, 'neural_network')


# Example usage:
# classification_model = Classification(link='your_dataset_link.csv')
# features_train, features_test, target_train, target_test = train_test_split(...your split logic...)
# metrics_tree = classification_model.decision_tree(features_train, target_train, features_test, target_test)
# metrics_knn = classification_model.knn(features_train, target_train, features_test, target_test)
# metrics_lr = classification_model.logistic_regression(features_train, target_train, features_test, target_test)
# metrics_rf = classification_model.random_forest(features_train, target_train, features_test, target_test)
# metrics_svm = classification_model.support_vector_machine(features_train, target_train, features_test, target_test)
# metrics_nn = classification_model.neural_network(features_train, target_train, features_test, target_test)
