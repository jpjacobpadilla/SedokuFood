from difflib import restore
from tracemalloc import start
from flask import Flask, render_template, url_for, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import re
import time
from project_secrets import user_db, password_db, host_db, name_db

app = Flask(__name__)

conn_string = 'mysql+pymysql://{user}:{password}@{host}:{port}/{db}?charset=utf8'.format(
    user = user_db, 
    password = password_db, 
    host = host_db, 
    port = 3306, 
    encoding = 'utf-8',
    db = name_db
)

app.config['SQLALCHEMY_DATABASE_URI'] = conn_string
db = SQLAlchemy(app)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


@app.route('/explore-our-database', methods=['GET'])
def exploreourdatabase():
    return render_template('exploreourdatabase.html')

@app.route('/explorezipcodeprocess', methods=['POST'])
def explorezipcodeprocess():
    is_valid_zip = False
    zip = request.form['zipcode']
   
    rest_count_output_q = f'''
    select count(*)
    from rest_local as rl
    inner join location as l on rl.local_id=l.local_id
    where l.zipcode = '{zip}';
    '''
    avg_price_output_q = f'''
    select round(avg(rld.price), 0)
    from rest_local_dish as rld
    inner join location as l on rld.local_id=l.local_id
    where l.zipcode = '{zip}';
    '''
    avg_rating_output_q = f'''
    select round(avg(r.rating), 1)
    from restaurant as r
    inner join rest_local as rl on r.rest_id=rl.rest_id
    inner join location as l on rl.local_id=l.local_id
    where l.zipcode = '{zip}';
    '''

    if not re.search(r'\D', str(zip)):
        rest_count_output = str(db.session.execute(rest_count_output_q).first()[0])
        avg_price_output = str(db.session.execute(avg_price_output_q).first()[0])
        avg_rating_output = str(db.session.execute(avg_rating_output_q).first()[0])

        if rest_count_output != 'None' and \
            avg_price_output != 'None' and \
            avg_rating_output != 'None':
            is_valid_zip = True

        return jsonify({
            'answer': is_valid_zip,
            'fullzip': request.form['zipcode'],
            'zp1': list(zip)[0],
            'zp2': list(zip)[1],
            'zp3': list(zip)[2],
            'zp4': list(zip)[3],
            'zp5': list(zip)[4],
            'rest_count_output': rest_count_output,
            'avg_price_output' : '$' + avg_price_output,
            'avg_rating_output': avg_rating_output
        })
    
    else:
        return jsonify({'answer': is_valid_zip,
                        'fullzip': request.form['zipcode']
                        })

@app.route('/dishfinderalgo', methods=['POST'])
def dishfinderalgo():
    #Set Funciton vars
    start_time = time.perf_counter()
    valid_resp = False

    #Get variables from post request
    user_budget = request.form['budget']
    days_eating_out = request.form['days_to_eat_out']
    discount_factor_raw = request.form['food_uq']
    user_zip_location = request.form['zipcode']
    user_distance_preference = request.form['travel_dist']

    american_rating = request.form['american']
    asian_rating = request.form['asian']
    european_rating = request.form['euro']
    bar_rating = request.form['bar']
    b_c_rating = request.form['b_c']
    middle_eastern_rating = request.form['me']
    healthy_rating = request.form['healthy']
    misc_rating = request.form['misc']
    ethnic_rating = request.form['ethnic']
    latin_american_rating = request.form['latin']
    seafood_rating = request.form['seafood']

    #Check to see if user inputs are valid
    valid_resp = True
    err_msg = 'Something'

    #Run Algo

    #Return data in json format
    dishes = ['d1ewewewewewew', 'd2', 'd3', 'wfewfwefwefwefwefwe', 'wfewfwefwfwffg', 'r3r3rerereffef', 'efefefefefe']
    prices = ['p1ewewewewew', 'p2', 'p3', 'wfewfwefwefwefwefwe', 'wfewfwefwfwffg', 'r3r3rerereffef', 'efefefefefe']
    restaurant_names = ['r1ewewe', 'r2', 'r3', 'wfewfwefwefwefwefwe', 'wfewfwefwfwffg', 'r3r3rerereffef', 'efefefefefe']
    addresses = ['a1ewewewewewewewewewew', 'a2', 'a3', 'wfewfwefwefwefwefwe', 'wfewfwefwfwffg', 'r3r3rerereffef', 'efefefefefe']

    time.sleep(5 - (time.perf_counter() - start_time))
    return jsonify ({'answer': valid_resp,
                    'error_msg': err_msg,
                    'dishes': dishes,
                    'prices': prices,
                    'restaurant_names': restaurant_names,
                    'addresses': addresses
                        })


@app.route('/dishfinder', methods=['GET'])
def dishfinder():
    return render_template('dishfinder.html')


if __name__ == '__main__':
    app.run(debug=True)