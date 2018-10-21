import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingClassifier
# from sklearn.cross_validation import train_test_split

from sklearn import preprocessing
from sklearn.utils import shuffle


def load_data(file_path, percentage):
    df = pd.read_csv(file_path)
    df['price'] = np.log1p(df['price'])
    nor_x = preprocessing.normalize(df.values)
    df1 = pd.DataFrame(nor_x)
    df1.columns = df.columns

    df1 = shuffle(df)
    x = df1.drop('price', axis=1).values
    # x = df1[['accommodates', 'bathrooms', 'bedrooms', 'beds']].values
    y = df1['price'].values

    split_pt = int(len(x) * percentage)
    x_train = x[:split_pt]
    y_train = y[:split_pt]
    x_test = x[split_pt:]
    y_test = y[split_pt:]

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    # x_train, y_train, x_test, y_test = load_data("df_tr.csv", 0.7)
    x_train = pd.read_csv('df_tr.csv', index_col=0)
    y_train = pd.read_csv('y_train.csv', index_col=0)
    x_test = pd.read_csv("df_test.csv", index_col=0)
    y_test = pd.read_csv("y_test.csv", index_col=0).values
    # model = GradientBoostingRegressor(n_estimators = 600, max_depth = 5, min_samples_split = 2, learning_rate = 0.1, loss = 'ls')

    # model = linear_model.LinearRegression()

    # model.fit(x_train, y_train)
    print(x_train.head())
    print(x_train.iloc[0,[1,2]])
    print(list(y_test))
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(x_train, y_train)
    a = regressor.score(x_test, y_test)
    print(a)
    y_pred = regressor.predict(x_test)
    # y_pred = model.predict(x_test)
    # print(model.score(x_test, y_test))
    # print(regressor.coef)

    # for i in range(len(y_test)):
    #	print("Expected: ", y_test[i], "Predicted: ", y_pred[i])
    # print(y_test.shape)
    # print(y_pred.shape)
    print("Mean square error: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
# df = pd.read_csv("dfsub.csv")

# print(df['price'].skew())

# plt.plot(df['beds'], df['price'].values)
# plt.xlim(-0.5,5)
# plt.show()

