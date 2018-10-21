from flask import *
# from flask_restplus import Resource, Api
# from flask_restplus import fields
# from flask_restplus import inputs
# from flask_restplus import reqparse
from flask import Flask, jsonify
from flask import request
from flask_restplus import Resource, Api
from flask_restplus import fields
from flask_restplus import inputs
from flask_restplus import reqparse
from flask import Flask, request, jsonify
import requests
import googlemaps
from price_predicting import price_predicting
import pandas as pd
import json
import random

app = Flask(__name__)
api = Api(app)

gmaps = googlemaps.Client(key='AIzaSyDpsVC3jIBeB72qisoJQLSf2FeLZLx0tfw')

search_template = {'location': None, 'capacity': 0, 'room_type': '', 'price' : 0}

template_input = {"Host_since" : "20160928",
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
            "Room_type": "Entire room"}

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

@api.route('/rent/predict')# This function predicts the price of property using certain parameters
class predict(Resource):
    def get(self):# predict price for a bunch of given parameters
        #print(requests.get())
        #print("lol")
        print('#'*40)
        from pprint import pprint
        #pprint(template_input)
        #pred_price = 40
        #params = json.loads()
        #with open('params.json') as json_data:
        #    params = json.load(json_data)
        #from pprint import pprint
        #params = self.params
        #print(params)
        pred_price = price_predicting(template_input)
        # param_dict = json.loads(r.content)
        # # print(param_dict)
        # location = request.args.get('location')
        #timePeriod = request.args.get('time')
        #my_data = {"location":location, "time" : timePeriod}
        print(pred_price)
        return {"price" :int(pred_price)}

    def post(self):# Get house specificatoin from landlord and return predicted price
        params = request.json
        from pprint import pprint
        self.params = params['location']
        pprint(params)
        loc = params['location']['suburb']
        geo_loc = gmaps.geocode('Kingsford, Sydney')[0]['geometry']['location']
        template_input['Location']={'latitude': geo_loc['lat'], 'longitude':geo_loc['lng']}
        template_input['Rooms']['Bathrooms'] = float(params['rooms']['bathrooms'])
        template_input['Rooms']['Bedrooms'] = float(params['rooms']['bedrooms'])
        template_input['Rooms']['Beds_no'] = float(params['rooms']['beds_no'])
        template_input['Capacity']['Guests_included'] = float(params['capacity'])
        template_input['Period']['Minimum_nights']= float(params['duration']['minimum_nights'])
        template_input['Period']['Maximum_nights'] = 100#float(params['duration']['maximum_nights'])
        template_input['Capacity']['Extra_people'] = float(params['additional_fee']['extra_fee'])
        template_input['Reviews']['Cleaning_fee'] = float(params['additional_fee']['cleaning_fee'])
        #self.params = template_input
        print('*'*40)
        pprint(template_input)
        #with open('params.json', 'w') as outfile:
        #    json.dump(template_input, outfile)
        #json.dump(template_input)

        #print("lel")
        #resp = request.get_json()#Get request object
        #price = price_predicting.price_predicting(resp)
        #sendObj = {"sellPrice": str(price)}
        return None

# @api.route('/rent')# This function predicts the price of property using certain parameters
# class predict(Resource):
#     def get(self):# predict price for a bunch of given parameters
#         r = requests.get()
#        	print(r)
#         return r


#############User query endpoint############
# for this endpoint, a tenant will search a room with 'loaction(suburb name)', 'accommodates', 'price'
# use an api to do the search



@api.route('/roomsearch')
class roomsearch(Resource):
    def get(self):# predict price for a bunch of given parameters
        #print(requests.get())
        #print("lol")
        print('#'*40)
        from pprint import pprint
        #pprint(template_input)
        df = pd.read_csv('df_cleaned.csv')
        location = search_template['location']
        locations = [location + ' , Australia', location + ', Australia']
        print(locations)
        dfloc = df[df['smart_location'].isin(locations)]
        df_res = dfloc[(dfloc['price'] < search_template['price']) & \
                       (dfloc['accommodates'] < search_template['capacity'])]
        print( '*'*20, df_res.shape)
        df_res2 = df_res[df_res['room_type'] == search_template['room_type']]
        print('&'*20, df_res2.shape)
        df_res3 = df_res2[['name','price']]
        pprint(df_res3.to_dict(orient = 'index'))
        return jsonify(df_res3.to_dict(orient = 'index'))

    def post(self):# Get house specificatoin from landlord and return predicted price
        params = request.json
        from pprint import pprint
        self.params = params['location']
        pprint(params)
        loc = params['location']['suburb']
        accom = float(params['capacity'])
        r_type = params['room_type']
        price = float(params['price'])
        search_template['location'] = loc
        search_template['room_type'] = r_type
        search_template['capacity'] = accom
        search_template['price'] = price
        #self.params = template_input
        print('*'*40)
        pprint(search_template)
        #with open('params.json', 'w') as outfile:
        #    json.dump(template_input, outfile)
        #json.dump(template_input)

        #print("lel")
        #resp = request.get_json()#Get request object
        #price = price_predicting.price_predicting(resp)
        #sendObj = {"sellPrice": str(price)}
        return None




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
        resualt = {'loc':location, 'avg_rent':int(mean_rent)}
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




########

@app.route("/login", methods=['GET', 'POST'])
def login():
	if request.method == "GET":
		return render_template("login.html")
	if request.method == "POST":
		print (request.form) 
		# this is the form:
		# ImmutableMultiDict([('username', 'Lucian'), ('password', '123456'), ('type', 'Guest')])
		return redirect("/landlord")
 

@app.route('/landlord', methods=['GET', 'POST'])
def landlord():
	return render_template("landlord.html")
	# if request.method == "GET":
	# 	return render_template("landlord.html")
	# if request.method == "POST":
	# 	return render_template("login.html")


@app.route('/user', methods=['GET', 'POST'])
def user():
	return render_template("user.html")
	# if request.method == "GET":
	# 	
	# if request.method == "POST":
	# 	return render_template("login.html")
 

@app.route('/specialist', methods=['GET', 'POST'])
def specialist():
	return render_template("specialist.html")
	# if request.method == "GET":
	# 	return render_template("login.html")
	# if request.method == "POST":
	# 	return render_template("login.html")



if __name__ == '__main__':
	app.run()
