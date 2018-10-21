import requests
#import data_processing
#import price_predicting
import pandas as pd
import json
from flask import Flask, jsonify
from flask import request
from flask_restplus import Resource, Api
from flask_restplus import fields
from flask_restplus import inputs
from flask_restplus import reqparse

def read_csv(csv_file):

    return pd.read_csv(csv_file)

def clean_csv(dfm):

    dfm = dfm[~dfm['rent'].isnull()]
    dfm['rent'] = dfm['rent'].astype(str)
    dfm['rent'] = dfm['rent'].str.replace(',', '')#clean data
    dfm['rent'] = dfm['rent'].str.replace(' ', '')#clean data
    dfm['rent'] = dfm['rent'].astype(float)
    return dfm

def get_location_names(dfm,location):
    #print(list(dfm.location.unique()))
    print(list(dfm['smart_location'].unique()))
    #if location not in list(list(dfm['smart_location'].unique())):
    #    return False
    #else:
    #    True
    flag = False
    for loc in list(dfm['smart_location'].unique()):
        if location.lower() in loc.lower():
            print(loc)
            flag = True

    return flag


def topBottom(dfm, type, number, location, roomtype):

    #dfm = dfm.loc[dfm['location'] == location]
    #dfm = dfm.sort_values(by=['rent'],ascending=False)
    print(type)
    locations = [location + ' , Australia', location + ', Australia']
    df_loc = dfm[dfm['smart_location'].isin(locations)]
    df_loc_rt = df_loc[df_loc['room_type'] == roomtype]
    print(df_loc_rt['room_type'].head())
    df_rank = df_loc_rt.sort_values(by=['price'], ascending=False)
    if type == 'top':
        rent = list(df_rank[:number]['price'])[0]
        #ids = list(dfm[:number].index)
    
    else:
        rent = list(df_rank[-number:]['price'])[0]
        #address = list(dfm[-number:]['address'])
    
    #my_data = dict(zip(address, rent))

    my_data = {"location": location, "list_house" :rent}
    return my_data

def inrange(dfm, maxx, minn, location):
    #dfm = dfm.loc[dfm['location'] == location]
    loc_list = ['Kingsford, Australia','Kingsford , Australia']
    df_loc = dfm[dfm['smart_location'].isin(loc_list)]
    df_rng = df_loc[(df_loc['price'] >= minn) & (df_loc['price'] <= maxx)]
    #print(dfm)
    rent = list(df_rng['price'])
    address = list(df_rng.index)
    df = df_rng.groupby(['property_type']).agg(['count']).iloc[:,0]
    #my_data = dict(zip(address, rent))
    my_data = {"location": location, "range_house" :df.to_dict()}

    return my_data

def getAvg(dfm, location):
    dfm = dfm.loc[dfm['location'] == location]
    total_rent = list(dfm['rent'])
    avg =  sum(total_rent)/len(total_rent)
    return {"location":location, "avg":avg}


app = Flask(__name__)
api = Api(app)

@api.route('/rent/predict')# This function predicts the price of property using certain parameters
class predict(Resource):
    def get(self):# predict price for a bunch of given parameters
        r = requests.get()
        param_dict = json.loads(r.content)
        location = request.args.get('location')
        #timePeriod = request.args.get('time')
        #my_data = {"location":location, "time" : timePeriod}
        return location

    #def post(self):# Get house specificatoin from landlord and return predicted price
        #resp = request.get_json()#Get request object
        #price = price_predicting.price_predicting(resp)
        #sendObj = {"sellPrice": str(price)}
        #return jsonify(sendObj)

#############User query endpoint############
# for this endpoint, a tenant will search a room with 'loaction(suburb name)', 'accommodates', 'price'
# use an api to do the search
@api.route('/roomsearch')




##############################################

@api.route('/avg/rent/<string:location>')
class average(Resource):
    def get(self, location):      
        #file = read_csv("rent_house.csv")
        dfm = pd.read_csv('df_cleaned.csv')
        #dfm = clean_csv(file)
        location_exist = get_location_names(dfm, location)
        if(location_exist is False):
            return jsonify({"message":f'{location} does not exist'})
        #my_data = getAvg(dfm, location)
        locations=[location+' , Australia', location+', Australia']
        print(locations)
        my_data = dfm[dfm['smart_location'].isin(locations)]
        mean_rent = my_data['price'].mean()
        resualt = {'loc':location, 'avg_rent':mean_rent}
        return jsonify(resualt)

@api.route('/rent/range/<string:location>') # this api returns the list of house in a location in the range(min, max)
class rangef(Resource):
    def get(self, location):
        maximum = int(request.args.get('max'))
        minimum = int(request.args.get('min'))
        #file = read_csv("rent_house.csv")
        dfm = pd.read_csv('df_cleaned.csv')
        #dfm = clean_csv(file)
        location_exist = get_location_names(dfm, location)
        if not location_exist:
            return jsonify({"message":f'{location} does not exist'})
        my_data = inrange(dfm, maximum, minimum, location)
        return jsonify(my_data)

@api.route('/rent/<string:location>') # this api returns the number of most or least expensive house in a location
class topbottom(Resource):
    def get(self, location):        
        typeof = request.args.get('typeof')
        number = int(request.args.get('number'))
        roomtype = request.args.get('roomtype')
        print(roomtype)
        if roomtype == 'entire':
            roomtype = 'Entire home / apt'
        elif roomtype == 'private':
            roomtype = 'Private room'
        elif roomtype == 'shared':
            roomtype = 'Shared room'
        dfm = read_csv("df_cleaned.csv")
        location_exist = get_location_names(dfm, location)
        if(location_exist is False):
            return jsonify({"message":f'{location} does not exist'})
        
        my_data = topBottom(dfm, typeof, number, location, roomtype)
        return jsonify(my_data)


if __name__ == '__main__':
    #run the application
    #dp = data_processing.Data_processor()
    #dp.data_processing()
    #dp.data_to_numeric()
    #dp.make_train_test()
    app.run(port = 5000, debug=True)

