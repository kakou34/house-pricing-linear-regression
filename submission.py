import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Function to get the data from a local file
def prepare_data(path):

    # Importing the data to a numpy array from a file
    dataset = pd.read_csv(path)
    # Extracting the columns of the features
    features = dataset.drop('price', 1)
    # Extracting the labels (prices)
    labels = dataset['price']

    # Splitting the data to 80% Training and 20% Test sets
    split_point = (int(round(len(labels) * 0.8)))
    train_features = features[:split_point]
    test_features = features[split_point:]
    train_labels = labels[:split_point]
    test_labels = labels[split_point:]

    return train_features, train_labels, test_features, test_labels


# Creating, Training and evaluating a Linear Regression model
def create_model(train_features, train_labels, test_features, test_labels):

    # Initializing the Linear Regression model
    regression = LinearRegression()

    # Training the model using the training set
    regression.fit(train_features, train_labels)

    # Making predictions using the testing set
    predictions = regression.predict(test_features)

    # Finding the evaluation metric RMSE
    rmse = math.sqrt(mean_squared_error(test_labels, predictions))
    print(regression.score(test_features,test_labels))

    return regression, rmse


if __name__ == '__main__':
    path = 'prices.csv'
    try:
        # Data preparation
        train_features, train_labels, test_features, test_labels = prepare_data(path)

        # Model training and evaluation
        regression, rmse = create_model(train_features, train_labels, test_features, test_labels)

        # Printing the results
        print('Model Coefficients: \n', regression.coef_)
        print("RMSE = {:.2f}".format(rmse))

    except FileNotFoundError:
        print("File not found! Please try again")
    except KeyError:
        print("Data format does not match the requirements!")
