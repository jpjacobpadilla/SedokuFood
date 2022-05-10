from decimal import Decimal
from distutils.log import error
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import torch
from flask import  jsonify
import pickle

# from geopy.geocoders import Nominatim
# import geopandas as gpd
# from matplotlib.figure import Figure
# import pgeocode
# import base64
# from io import BytesIO


NN_model = pickle.load(open('model.pkl', 'rb'))

def filter_location(df, distance_df, user_zip_location, distance):
    df_output = df.copy()
    inrange_zip_list = []

    for i, v in distance_df[user_zip_location].items():
        if v <= distance:
            inrange_zip_list.append(i[0])

    query = df_output.zipcode.isin(inrange_zip_list)
    df_output = df_output[query]

    return df_output


def filter_cuisine(df, reccomended_cuisine):
    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df.cuisine_clusters == reccomended_cuisine]
    return filtered_df


def get_restaurant_all_dishes(df, all_dishes):
    one_restaurant_all_dishes = df.merge(all_dishes, on = 'rest_id', how = 'inner')
    return one_restaurant_all_dishes


def filter_dish_price(df, max_daily_budget):
    min_daily_budget = max_daily_budget/1.5

    query1 = df.price <= max_daily_budget
    df = df[query1]
    
    query2 = df.price >= min_daily_budget
    df = df[query2]
    return df


def dish_rec_main(request, db):
    #Get variables from post request and check user-input
    try:
        user_budget = int(request.form['budget'])
    except:
        return jsonify ({'answer': False,
        'error_msg': 'Uh Oh! Please fix the budget input field.'}) 

    user_zip_location = str(request.form['zipcode'])
    # Check if zipcode is all ints
    if user_zip_location.isnumeric() is False:
        err_msg = 'Uh Oh! Please enter only numbers for the zipcode field.'
        return jsonify ({'answer': False,
        'error_msg': err_msg}) 

    try:
        days_eating_out = int(request.form['days_to_eat_out'])
        discount_factor_raw = int(request.form['food_uq'])
        user_distance_preference = int(request.form['travel_dist'])
        input_splurge_func = float(request.form['input_splurge_func'])
        american_rating = int(request.form['american'])
        asian_rating = int(request.form['asian'])
        european_rating = int(request.form['euro'])
        bar_rating = int(request.form['bar'])
        b_c_rating = int(request.form['b_c'])
        middle_eastern_rating = int(request.form['me'])
        healthy_rating = int(request.form['healthy'])
        misc_rating = int(request.form['misc'])
        ethnic_rating = int(request.form['ethnic'])
        latin_american_rating = int(request.form['latin'])
        seafood_rating = int(request.form['seafood'])
    except:
        return jsonify ({'answer': False,
        'error_msg': 'Uh Oh! Something went wrong :('}) 

    # Check if zipcode is in database
    query_string = f'''
    select l.local_id
    from location as l
    where l.zipcode = '{user_zip_location}';
    '''

    try:
        query_result = db.session.execute(query_string).fetchall()
    except:
        err_msg = 'Uh Oh! Something went wrong while verifing your inputs :('
        return jsonify ({'answer': False, 'error_msg': err_msg}) 

    if len(query_result) == 0:
        err_msg = 'Uh Oh! That zipcode is not in our database :('
        return jsonify ({'answer': False, 'error_msg': err_msg}) 

    # User can only splurge if days eating out is more than or equal to 3
    if days_eating_out <= 2 and input_splurge_func != 0: #splurge has to be 0
        err_msg = 'Uh Oh! You cannot splurge if you eat out less than 3 days  :( '
        return jsonify ({'answer': False,
        'error_msg': err_msg}) 

    # Make sure price is in a specific range
    budget_per_day = user_budget / days_eating_out
    if budget_per_day < 10:
        err_msg = 'Uh Oh! You cannot have an average budget of less that 10 dollars per day'
        return jsonify ({'answer': False,
        'error_msg': err_msg}) 
    elif budget_per_day > 85:
        err_msg = 'Uh Oh! You cannot have an average budget of more than 85 dollars per day.'
        return jsonify ({'answer': False,
        'error_msg': err_msg}) 

    # Catch-all try and except block
    try: 
        # Start of Model
        input_rating = [american_rating, asian_rating ,european_rating, bar_rating, b_c_rating, middle_eastern_rating,
                healthy_rating, misc_rating,  ethnic_rating, latin_american_rating, seafood_rating]
        input_rating = torch.FloatTensor(input_rating)

        prob_list = NN_model(input_rating)

        cuisine_list= ['American', 'Asian', 'European', 'Bar',
                'Bakery/Cafe', 'Middle Eastern', 'Vegan/Healthy',
                'Miscellaneous',
                'Ethnic Food', 'Latin American', 'Seafood']
        cuisine_rec = []

        #NN algo needs to update and apply discount factor every time it chooses one cuisine.
        for i in range(days_eating_out):
            rec_index = torch.max(prob_list, dim =0)[1]
            cuisine_rec.append(cuisine_list[rec_index])
            prob_list[rec_index] *= (discount_factor_raw*0.1)

        all_restaurant_query = pd.read_sql_query('''
        select r.rest_id, r.name, r.rating, r.cuisine, r.price_range, cuisine_clusters, l.address, l.zipcode
        from restaurant as r
        inner join rest_local as rl on rl.rest_id=r.rest_id
        inner join location as l on l.local_id=rl.local_id;
        ''', db.session.bind)
        distance_df = pd.read_sql_query ('''
                                        SELECT * from distance_df
                                        ''', db.session.bind)
        distance_df.index = [distance_df.columns]

        all_dishes = pd.read_sql_query ('''
                                        SELECT * from rest_local_dish
                                        ''', db.session.bind)
        all_dishes = all_dishes[['rest_id', 'dish_name', 'price']]

        #connected to splurge_user
        #can't splurge if days_eating_out is less than or equal to 2 days.
        splurge_num_days = int(days_eating_out*input_splurge_func)
        splurge_list = [False] * days_eating_out
        for i in range(splurge_num_days):
            splurge_list[i] = True

        remaining_budget = user_budget
        remaining_days_eating_out = days_eating_out
        final_dish_df = pd.DataFrame()
        picked_restaurant_rest_id = []

        filtered_location = filter_location(all_restaurant_query, distance_df, user_zip_location, user_distance_preference)

        for i, cuisine in enumerate(cuisine_rec):
            filtered_cuisine = filter_cuisine(filtered_location, cuisine)
            merged_restaurant_dishes = get_restaurant_all_dishes(filtered_cuisine, all_dishes) #merge restaurant with dishes
            max_daily_budget = remaining_budget/remaining_days_eating_out #change prices
            ##splurge
            if splurge_list[i] == True:
                filtered_price = filter_dish_price(merged_restaurant_dishes, max_daily_budget * 2)
            else:
                filtered_price = filter_dish_price(merged_restaurant_dishes, max_daily_budget) #filter by price

            top_rating = filtered_price.nlargest(int(len(filtered_price)/5), 'rating') # subsample top 20% highest rating dish/restaurants
            top_rating.drop(top_rating[top_rating['rest_id'].isin(picked_restaurant_rest_id)].index,inplace = True) #remove picked restaurants from dataset.

            picked_dish = top_rating.sample() # pick random dish
            final_dish_df = final_dish_df.append([picked_dish])
            picked_restaurant_rest_id.append(picked_dish.rest_id) # to make sure restaurant is not picked again

            remaining_budget -= float(picked_dish.price)
            remaining_days_eating_out -= 1

        final_dish_df.reset_index(drop = True, inplace = True)

        # Check to make sure there is an output
        if len(list(final_dish_df.dish_name)) == 0:
            return ({'answer': False,
                 'error_msg': 'Uh Oh! Our neural network could not find any dishes. Try changing your input.'})

        #Return data in json format
        addresses = list(final_dish_df.address)
        return jsonify ({'answer': True,
                        'error_msg': '',
                        'dishes': list(final_dish_df.dish_name),
                        'prices': list(final_dish_df.price),
                        'restaurant_names': list(final_dish_df.name),
                        'addresses': addresses,
                        'location_map': location_map(addresses)
                            })
    
    except Exception as e:
        err_msg = "Uh Oh! Something happend :(  -  Please try again."
        return ({'answer': False,
                 'error_msg': err_msg})

def location_map(addresses):
    #getting coordinates of our locations
    geolocator = Nominatim(user_agent="school project app")
    x_coordinates = []
    y_coordinates = []
    for address in addresses:
        geodata = geolocator.geocode(f'{address}, New York City, NY')
        y_coordinates.append(geodata.raw.get("lat"))
        x_coordinates.append(geodata.raw.get("lon"))

    x_coordinates = [float(x) for x in x_coordinates]
    y_coordinates = [float(y) for y in y_coordinates]

    nomi = pgeocode.Nominatim('us')
    user_zip_query = nomi.query_postal_code(user_zip_location)

    map_df = pd.DataFrame({'LONGITUDE' : x_coordinates, 'LATITUDE': y_coordinates})

    #creating base map of df_nyc
    df_nyc = gpd.GeoDataFrame.from_file('nyc-neighborhoods.geojson')
    base = df_nyc.plot(linewidth=0.5, color='White',edgecolor = 'Grey', figsize = (15,10))
    map_plot = map_df.plot (kind='scatter', 
        x = 'LONGITUDE', y = 'LATITUDE',
        figsize = (10, 7.5),
        s = 30, alpha = 1, color = [(248/256, 110/256, 81/256)], ax = base, edgecolor='face') #red
    
    fig = Figure()
    map_plot = fig.scatter(x = user_zip_query['longitude'], y = user_zip_query['latitude'], s = 30, color = 'green') #[(249/256,221/256,112/256)])
    map_plot.axis('off')
    
    # Save it to a temporary buffer
    buf = BytesIO()
    fig.savefig(buf, format="png")

    return base64.b64encode(buf.getbuffer()).decode("ascii")
    # https://matplotlib.org/3.5.0/gallery/user_interfaces/web_application_server_sgskip.html