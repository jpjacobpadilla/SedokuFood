import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import model_selection
from IPython import display


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
    valid_resp = False

    #Get variables from post request
    user_budget = int(request.form['budget'])
    days_eating_out = int(request.form['days_to_eat_out'])
    discount_factor_raw = int(request.form['food_uq'])
    user_zip_location = str(request.form['zipcode'])
    user_distance_preference = int(request.form['travel_dist'])
    input_splurge_func = int(request.form['input_splurge_func'])

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

    #Check to see if user inputs are valid
    valid_resp = True
    err_msg = 'Something'

    #Run Algo (put backend code here)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = pd.read_sql_query ('''
                            SELECT * from training_data
                            ''', db.session.bind)
    df= df[['American Food', 'Asian Food', 'European Food', 'Bar Food',
            'Bakery/Cafe Food', 'Middle Eastern Food', 'Healthy Food',
            'Miscellaneous Food + activities (e.g. food truck bazaars)',
            'Ethnic Food (e.g. Somalian Food)', 'Latin American Food', 'Seafood',
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']]
    df.replace({'No': 0, 'Yes': 1}, inplace = True)

    X = df[['American Food', 'Asian Food', 'European Food', 'Bar Food',
            'Bakery/Cafe Food', 'Middle Eastern Food', 'Healthy Food',
            'Miscellaneous Food + activities (e.g. food truck bazaars)',
            'Ethnic Food (e.g. Somalian Food)', 'Latin American Food', 'Seafood']]
    y = df[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']]

    X = torch.tensor(X.values).type(torch.FloatTensor).to(device)
    y = torch.tensor(y.values).type(torch.FloatTensor).to(device)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,test_size = 0.1, random_state = 100)
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    learning_rate = 5e-3

    D = 11  # dimensions
    C = 11  # num_classes
    H = 50  # num_hidden_units

    NN = nn.Sequential(
        nn.Linear(D,H),
        nn.ReLU(),
        nn.Linear(H,C),
        nn.Sigmoid()
    )
    NN.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(NN.parameters(), lr=learning_rate, momentum=0.5)
    for t in range(2000):
            y_pred = NN(X_train)
            loss = criterion(y_pred, y_train)
            display.clear_output(wait=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    #---#
    input_rating = [american_rating, asian_rating ,european_rating, bar_rating, b_c_rating, middle_eastern_rating,
            healthy_rating, misc_rating,  ethnic_rating, latin_american_rating, seafood_rating]
    input_rating = torch.FloatTensor(input_rating)
    prob_list = NN(input_rating)
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
    #---#
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


    #---#
    remaining_budget = user_budget
    remaining_days_eating_out = days_eating_out
    final_dish_df = pd.DataFrame()
    picked_restaurant_rest_id = []

    filtered_location = filter_location(all_restaurant_query, distance_df, user_zip_location, user_distance_preference)

    for cuisine in cuisine_rec:
        filtered_cuisine = filter_cuisine(filtered_location, cuisine)
        merged_restaurant_dishes = get_restaurant_all_dishes(filtered_cuisine, all_dishes) #merge restaurant with dishes
        max_daily_budget = remaining_budget/remaining_days_eating_out#change prices

        filtered_price = filter_dish_price(merged_restaurant_dishes, max_daily_budget) #filter by price

        top_rating = filtered_price.nlargest(int(len(filtered_price)/5), 'rating') # subsample top 20% highest rating dish/restaurants
        top_rating.drop(top_rating[top_rating['rest_id'].isin(picked_restaurant_rest_id)].index,inplace = True) #remove picked restaurants from dataset.

        picked_dish = top_rating.sample() # pick random dish
        final_dish_df = final_dish_df.append([picked_dish])
        picked_restaurant_rest_id.append(picked_dish.rest_id) # to make sure restaurant is not picked again

        remaining_budget -= float(picked_dish.price)
        remaining_days_eating_out -= 1

    final_dish_df.reset_index(drop = True, inplace = True)
    #---#
    #Return data in json format
    dishes = list(final_dish_df.dish_name)
    prices = list(final_dish_df.price)
    restaurant_names = list(final_dish_df.name)
    addresses = list(final_dish_df.address)

    return dishes, prices, restaurant_names, addresses, valid_resp, err_msg