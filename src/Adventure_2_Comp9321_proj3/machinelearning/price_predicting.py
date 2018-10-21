import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.utils import shuffle
from sklearn import preprocessing
import json
from data_processing import Data_processor

###########TEST#########
test_input = {"Host_since" : "20160928",
            "Location" : {
                            "latitude": -33.87205560158492,
                            "longitude": 151.222701315151,
                            } ,
            "Rooms" : {
                        "Bathrooms" : 2.0,
                        "Bedrooms" : 1.0,
                        "Beds_no" : 1.0,
                        },

            "Capacity" : {
                            "Guests_included" : 2,
                            "Extra_people" : 0,
                            },
            "Period" :{
                            "Minimum_nights" : 4.0,
                            "Maximum_nights" : 1125
                        },
            "Reviews" : {
                            "Cleaning_fee" : 0
                        },
            "Room_type": "Entire room"
              }
########################


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

def supervised_train(dummy = False):
    '''
    This function will train our dataset with a regression model.
    :param file: File name for the necessary dataset
    :param dummy: When it is False(defualt) we will only use numeric data for training, otherwise we use numeric and
            dummy variables.
    :return: A Regression model
    '''
    X_train = pd.read_csv('df_tr.csv', index_col=0)
    Y_train = pd.read_csv('y_train.csv', index_col=0).values
    print(X_train.shape)
    print(Y_train.shape)
    if dummy:
        rfr = RandomForestRegressor(n_estimators=100, random_state=0, \
                                    oob_score=True, n_jobs=-1)
    else:
        rfr = RandomForestRegressor(n_estimators=100, random_state=0, \
                                   oob_score=True, n_jobs=-1)
    rfr.fit(X_train, Y_train)
    print(rfr.oob_score_)
    return rfr

def unsupervised_learn(entry, data, dummy = True):
    '''

    :param dummy:
    :return:
    '''
    dp = Data_processor(read = False)
    print(entry.columns)
    entry = entry.drop(['cleaning_fee', 'extra_people', 'maximum_nights'], axis=1)
    pred = dp.data_embedding(data = data, entry=entry)
    return pred

def input_processing(input):
    '''

    :param input: Json file
    :return: an array with different input
    '''
    #para_dict = json.loads(input)
    para_dict = {'host_since':0,'latitude':0, 'longitude':0, 'accommodates':0, 'bathrooms':0,
                 'bedrooms':0, 'beds':0,'cleaning_fee':0, 'extra_people':0,
                 'minimum_nights':0, 'maximum_nights':0, 'entire':0, 'private':0, 'shared':0}
    para_dict['host_since'] = float(input['Host_since'])
    para_dict['latitude'] = input['Location']['latitude']
    para_dict['longitude'] = input['Location']['longitude']
    para_dict['accommodates'] = input['Capacity']['Guests_included']
    para_dict['bathrooms'] = input['Rooms']['Bathrooms']
    para_dict['bedrooms'] = input['Rooms']['Bedrooms']
    para_dict['beds'] = input['Rooms']['Beds_no']
    para_dict['cleaning_fee'] = input['Reviews']['Cleaning_fee']
    para_dict['extra_people'] = input['Capacity']['Extra_people']
    para_dict['minimum_nights'] = input['Period']['Minimum_nights']
    para_dict['maximum_nights'] = input['Period']['Maximum_nights']
    if 'Entire' in input['Room_type']:
        para_dict['entire'] = 1
    elif 'Private' in input['Room_type']:
        para_dict['private'] = 1
    elif 'Shared' in input['Room_type']:
        para_dict['shared'] = 1


    input = pd.DataFrame(para_dict, index = [0])
    return  input

def price_predicting(input, learn_mode = 'mixed', dummy = True, train_mode = True, normalized = False):
    '''
    This function
    :param input: A data input with features in a dictionary format need to return a predicitng price
    :param learn_mode: This parameter take a string input. It can either be 'unsupervised', 'supervised'
            or 'mixed'(default). If it's 'unsupervised' this function will use clustering model and take average;
            if it's surpervised this function will use regression model and predict price; if it's mixed will take
            a weighted average price from both methods.
    :param dummy: When it is False(defualt) we will only use numeric data for training, otherwise we use numeric and
            dummy variables.
    :param train_mode: When this is True(default) we will train our model and save it. When learn_mode has been set as
                        unsupervised this train_mode will be ignored.
    :return: A predicting price
    '''

    input = input_processing(input)
    if learn_mode == 'mixed' and train_mode:
        if dummy:
            rfr = supervised_train(dummy=True)
        else:
            rfr = supervised_train()
        pickle.dump(rfr, open('RandomForrestRegressor.txt','wb'))
        x_test = pd.DataFrame(input)
        if not normalized:
            pred_sup = rfr.predict(x_test)
        else:
            #TODO: Process unnormalized input to train an normalized data
            pass

        #pred_up = unsupervised_learn(input)
        df = pd.read_csv('df_dummy_numeric.csv', index_col=0)
        pred_up = unsupervised_learn(input, df)
        pred = pred_sup * 0.3 + pred_up * 0.7
        return pred[0]
    elif learn_mode == 'mixed' and not train_mode:
        rfr = pickle.load(open('RandomForrestRegressor.txt', 'r'))
        x_test = pd.DataFrame(input)
        if not normalized:
            pred_sup = rfr.predict(x_test)
        else:
            pass

        pred_up = unsupervised_learn(input)
        pred = pred_sup * 0.4 + pred_up * 0.6
        return pred

    elif learn_mode == 'supervised' and train_mode:
        if dummy:
            rfr = supervised_train(dummy=True)
        else:
            rfr = supervised_train()
        rfr = supervised_train()
        pickle.dump(rfr, open('RandomForrestRegressor.txt', 'wb'))
        x_test = pd.DataFrame(input)
        if not normalized:
            pred_sup = rfr.predict(x_test)
        else:
            # TODO: Process unnormalized input to train an normalized data
            pass
        return pred_sup

    elif learn_mode == 'supervised' and not train_mode:
        rfr = pickle.load(open('RandomForrestRegressor.txt', 'r'))
        x_test = pd.DataFrame(input)
        if not normalized:
            pred_sup = rfr.predict(x_test)
        else:
            pass

        return pred_sup

    elif learn_mode == 'unsupervised':
        df = pd.read_csv('df_dummy_numeric.csv', index_col=0)
        pred = unsupervised_learn(input, df)
        return pred[0]



if __name__ == '__main__':
    from sklearn.metrics import mean_squared_error
    #rfr = supervised_train()
    #df_test = pd.read_csv('df_test.csv', index_col=0)
    #y_test = pd.read_csv('y_test.csv', index_col=0)
    #y_pred = rfr.predict(df_test)
    #me = np.sqrt(mean_squared_error(y_pred, y_test))
    pred = price_predicting(input = test_input, learn_mode='mixed')
    print(pred)




