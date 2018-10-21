import pandas as pd
import numpy as np
import sklearn
import random
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors

class Data_processor():
    def __init__(self, file = 'listings.csv', read = True):
        self.file = file
        if read:
            self.df = pd.read_csv(file)

    def data_processing(self):
        df = self.df
        #select useful columns
        df2 = df[['id', 'host_since', 'host_response_time', \
                  'host_response_rate', 'host_is_superhost', \
                  'host_verifications', 'smart_location', 'latitude', 'longitude', \
                  'is_location_exact', 'property_type', 'room_type', 'accommodates', \
                  'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities',
                  'square_feet', 'price', 'security_deposit', 'cleaning_fee',
                  'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights',
                  'number_of_reviews', 'review_scores_rating', 'cancellation_policy']]

        # Convert column values and fill Nan values
        df2['price'] = df2['price'].apply(lambda x: x[1:-3])
        df2['price'] = df2['price'].str.replace(',', '')
        df2['price'] = df2['price'].apply(pd.to_numeric)
        df2['extra_people'] = df2['extra_people'].fillna('$0.00')
        df2['extra_people'] = df2['extra_people'].apply(lambda x: x[1:-3])
        df2['extra_people'] = df2['extra_people'].str.replace(',', '')
        df2['extra_people'] = df2['extra_people'].apply(pd.to_numeric)
        df2['cleaning_fee'] = df2['cleaning_fee'].fillna('$0.00')
        df2['cleaning_fee'] = df2['cleaning_fee'].apply(lambda x: x[1:-3])
        df2['cleaning_fee'] = df2['cleaning_fee'].str.replace(',', '')
        df2['cleaning_fee'] = df2['cleaning_fee'].apply(pd.to_numeric)
        df2['security_deposit'] = df2['security_deposit'].fillna('$0.00')
        df2['security_deposit'] = df2['security_deposit'].apply(lambda x: x[1:-3])
        df2['security_deposit'] = df2['security_deposit'].str.replace(',', '')
        df2['security_deposit'] = df2['security_deposit'].apply(pd.to_numeric)
        # df2['accommodates'] = df2['accommodates'].apply(pd.to_numeric)
        df2['host_response_rate'] = df2['host_response_rate'].fillna('0%')
        df2['host_response_rate'] = df2['host_response_rate'].apply(lambda x: x[:-1])
        df2['host_response_rate'] = df2['host_response_rate'].apply(pd.to_numeric) / 100
        df2['host_since'] = df2['host_since'].str.replace('-', '').apply(pd.to_numeric)

        # Fill missing text type value with 'Unknown'
        text_cols = list(df2.select_dtypes(include=['object']).columns)
        print(text_cols)
        for c in text_cols:
            df2[c] = df2[c].fillna('Unknown')

        # Drop outliers
        places = df2['smart_location'].value_counts().to_dict()
        total = 0
        #ALL = df2.shape[0]
        sub_list = []
        for key, item in places.items():
            total += item
            if item <= 20:
                break
            sub_list.append(key)
        # drop entries which has a uncommon geo location
        df5 = df2.loc[df2['smart_location'].isin(sub_list)]
        # drop entries which has unlike price
        df5 = df5.loc[(df5['price'] <200) & (df5['price'] >= 5)]
        # Save cleaned version of data
        self.df_cleaned = df5
        df5.to_csv('df_cleaned.csv')
        return True


    def data_to_numeric(self, numeric_fea=None):
        '''
        This function will take out numeric features from dataset and normalized it.
        :param: numeric_fea: A list of numeric features we need to use in our model
        :return: True
        '''
        df3 = self.df_cleaned.drop(['amenities', 'host_verifications', 'square_feet','id'], axis=1)
        text_cols = list(df3.select_dtypes(include=['object']).columns)
        df3 = df3.drop(text_cols, axis=1)
        df3 = df3.fillna(0)
        df3p = df3['price'].tolist()
        df3_cp = df3
        df3 = df3.drop(['price'], axis=1)
        #df3_cp = df3_cp.drop(['price'], axis=1)

        from sklearn import preprocessing
        data_matrix = df3.values
        cols = list(df3.columns)
        normalized_data = preprocessing.scale(data_matrix, axis=0)
        dfsub = pd.DataFrame(normalized_data, columns=cols)
        dfsub['price'] = pd.Series(df3p)
        #drop unecessary cols:
        if numeric_fea:
            dfsub = dfsub[numeric_fea]
        else:
            #print(list(df3_cp.columns))
            fea_list = ['host_since', 'latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms', \
             'beds', 'price', 'cleaning_fee', 'extra_people', 'minimum_nights', 'maximum_nights']
            #dfsub = dfsub[['accommodates', 'bathrooms', 'bedrooms', 'beds', 'price', \
             #          'minimum_nights', 'maximum_nights', 'longitude', 'latitude']]
            dfsub = dfsub[fea_list]
            #dfsub['latitude'] = (dfsub['latitude'] - 33)* 100000000
            #dfsub['longitude'] = (dfsub['longitude']- 151) * 100000000
            #df3_cp = df3_cp[['accommodates', 'bathrooms', 'bedrooms', 'beds', 'price',\
             #          'minimum_nights', 'maximum_nights', 'longitude', 'latitude']]
            df3_cp = df3_cp[fea_list]
            #df3_cp['latitude'] = (df3_cp['latitude'] - 33) * 1000000
            #df3_cp['longitude'] = (df3_cp['longitude'] - 151) * 1000000

            dfsub_unormalized = pd.DataFrame(df3_cp.values, columns=list(df3_cp.columns))

        #Save the numeric features
        self.df_numeric = dfsub
        dfsub.to_csv('df_numeric.csv')
        self.df_numeric_unormalized = dfsub_unormalized
        dfsub_unormalized.to_csv('df_numeric_unormalized.csv')
        return True


    def numeric_and_dummy(self, type_only=True):
        '''
        This function will make a data set which concate numeric features and dummy features
        :param:loc_only: If loc is True(default) we will only use location dummy features, otherwise
                        we will include other useful dummy features
        :return: True
        '''
        dfsub = self.df_numeric_unormalized
        df3 = self.df_cleaned.drop(['amenities', 'host_verifications', 'square_feet', 'id'], axis=1)
        if type_only:
            #['property_type', 'room_type']
            dummy_df = pd.get_dummies(df3[['room_type']])

        else:
            pass

        npsub = np.concatenate((dfsub.values, dummy_df.values), axis=1)
        dfsub = pd.DataFrame(npsub, columns=list(dfsub.columns) + list(dummy_df.columns))
        dfsub = dfsub.fillna(0)
        self.df_dummy_numeric = dfsub
        dfsub.to_csv('df_dummy_numeric.csv')
        return True



    def make_train_test(self, dummy=False, unormalized = True):
        '''
        This function will make the train_test dataset for training
        :param dummy: If dummy set as False(default) we are only gonna use numeric features
        :return: True
        '''
        if not dummy and not unormalized:
            dfsub = self.df_numeric
        elif not dummy and unormalized:
            dfsub = self.df_numeric_unormalized
        elif dummy:
            dfsub = self.df_dummy_numeric

        ind_all = set(range(dfsub.shape[0]))
        df_tr = dfsub.sample(frac=0.8)
        frac_set = set(df_tr.index)
        rest_set = ind_all - frac_set
        rest_ind = list(rest_set)
        print(df_tr.shape)
        print(dfsub.shape)
        y_train = pd.DataFrame(dfsub.iloc[df_tr.index, :]['price'])
        df_test = dfsub.iloc[rest_ind, :]
        y_test = pd.DataFrame(dfsub.iloc[rest_ind, :]['price'])
        df_tr = df_tr.drop(['price'], axis=1)
        df_test = df_test.drop(['price'], axis=1)
        #keep all the train test dataset
        self.X_train = df_tr
        df_tr.to_csv('df_tr.csv')
        self.Y_train = y_train
        print(df_tr.shape)
        print(y_train.shape)
        y_train.to_csv('y_train.csv')
        self.X_test = df_test
        df_test.to_csv('df_test.csv')
        self.Y_test = y_test
        y_test.to_csv('y_test.csv')

        return  True

    def data_embedding(self, data, entry=None, topK=6, fea=None, entry_flag = True):
        '''
        This function will embed the data to a vector space.
        Exploit_mode is on when there is no match listing for current parameters. In such case we will only do the embedding
        with the non-empty features which we extract from para_dict. This function will receive a pseduo entry, and use that
        entry to find the alternatives.
        :param df: A dataframe past from main function. This df is directly read from db, hasn't been preprocessed yet
                    In exploit_mode, it will append an extra pseudo_entry
        :param fea: In exploit_mode this are the features which extract from the para_dict.
        :param topK: numbers of features to return, default value is 5.
        :param entry_id: The id of the property we want to find the most similar alternatives.
        :return: A dataframe contains neareast topK entries .
        '''
        #pre_set_fea = ['host_since', 'latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms', \
         #    'beds', 'price', 'cleaning_fee', 'extra_people', 'minimum_nights', 'maximum_nights']
        '''if exploit_mode:  # In exploit_mode we only use the features which has been included in the para_dict
            new_fea = ['id']
            new_fea2 = [x for x in pre_set_fea if x in fea]
            new_fea += new_fea2
            new_fea.append('review_scores_rating')
            assert 'id' in new_fea
        else:
            new_fea = pre_set_fea'''
        print(data.shape)
        dfnew = data
        dfnew = dfnew.fillna(0)  # fill in NAN missing data
        dfnew = dfnew.drop(['price', 'cleaning_fee', 'extra_people', 'maximum_nights'], axis = 1)
        cols = list(dfnew.columns)  # get the colum names except 'id'
        #data_matrix = dfnew.iloc[:, list(range(1, len(new_fea)))].values  # get the matrix form data for scaling
        data_matrix = dfnew.values
        normalized_data = preprocessing.scale(data_matrix, axis=0)  # get the rescaled data
        dfsub = pd.DataFrame(normalized_data, columns=cols)  # build a dataframe for the data
        embedding = dfsub.values  # save the scaled data in matrix form
        '''dfsub['id'] = pd.Series(
            dfnew['id'].tolist())  # match the old ids, since the old index doesn't work,creat new Series
        dfsub = dfsub.set_index('id')  # set id as index and covert to a dict
        dict_embed = dfsub.to_dict(orient='index')  # convert df to dict = {id: {col1:val1, col2:val2 ...}}
        for index, values in dict_embed.items():  # reshape the dict to the form {id: [vec_of_values]}
            val_list = []
            for index2, val2 in values.items():
                val_list.append(val2)
            dict_embed[index] = np.array(val_list)'''
        nn_search = NearestNeighbors(6, metric='euclidean')  # embedding the data into a high-dim space
        nn_search.fit(embedding)  # fit the data matrix
        #index_embedding = []  # list for representing range_id to 'id'
        #range_index = list(range(dfnew.shape[0]))
        #id_index = dfnew['id'].tolist()
        #for i in range_index:
        #    index_embedding.append(id_index[i])
        if not entry_flag:
            center = [dfsub.iloc[-1,:].values]
        else:
            center = entry
        distance, neighbors = nn_search.kneighbors( center,
                                                   topK)  # find the nearst neighbors for entry_id.
        ##########Test#######
        print(distance, neighbors)
        #print('pirce: ', data.iloc[-1,:]['price'])
        #print(list(dfnew.columns))
        prices =[]
        for ind in neighbors[0][1:]:
            #print('nprice: ', data.iloc[ind, :]['price'])
            prices.append(data.iloc[ind, :]['price'])

        distance_norm = distance[0][1:]/sum(distance[0][1:])
        #print(distance_norm)
        distance_weighted_price = np.array(prices) * np.array(distance_norm)
        sum_of_distance_wprice = sum(distance_weighted_price)
        distance_weighted_ave_price = sum_of_distance_wprice
        print(distance_weighted_ave_price)
        return distance_weighted_ave_price

        '''scores = []
        for j in range(len(neighbors[0][1:])):
            id = neighbors[0][j + 1]
            score = 100 - round(distance[0][j + 1] * 100)
            scores.append((id, score))
            # scores['scores_of_'+str(id)] =
        scores.sort()
        print(scores)
        #########End#######
        result = df.iloc[neighbors[0][1:], :]  # A narrowed topK most similar rows dataframe with the original cols
        scores_list = [sc for id, sc in scores]
        result['scores'] = scores_list
        result = result.sort_values(by=['scores'], ascending=False)
        # pd.DataFrame.sort_values()
        id_list = result['id'].tolist()
        name_list = result['name'].tolist()
        neibour_list = result['neighbourhood'].tolist()
        accom_list = result['accommodates'].tolist()
        price_list = result['price'].tolist()
        min_nights = result['minimum_nights'].tolist()
        max_nights = result['maximum_nights'].tolist()
        url_list = result['picture_url'].tolist()
        property_list = result['property_type'].tolist()
        room_list = result['room_type'].tolist()
        scores_list = result['scores'].tolist()

        listings = []
        if topK > result.shape[0]:
            topK = result.shape[0]
        for i in range(topK):
            temp_dict = {"name": name_list[i],
                         "id": id_list[i],
                         "neighbourhood": neibour_list[i],
                         "accommodates": accom_list[i],
                         "price": price_list[i],
                         "minimum_nights": min_nights[i],
                         "maximum_nights": max_nights[i],
                         "picture_url": url_list[i],
                         "property_type": property_list[i],
                         "room_type": room_list[i]}
            # "recommend_score": scores_list[i]}
            listings.append(temp_dict)

        return listings'''




if __name__ == '__main__':
    dp = Data_processor()
    #dp.data_processing()
    #dp.data_to_numeric()
    #dp.numeric_and_dummy()
    #dp.make_train_test(dummy=True)
    df = pd.read_csv('df_dummy_numeric.csv', index_col=0)
    dp.data_embedding(data = df)

