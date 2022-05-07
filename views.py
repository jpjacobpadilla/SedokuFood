from difflib import restore
from tracemalloc import start
from flask import Flask, render_template, url_for, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import re
import time
from project_secrets import user_db, password_db, host_db, name_db
import dish_recommendor


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
    dishes, prices, restaurant_names, addresses, valid_resp, err_msg = dish_recommendor.dish_rec_main(request, db)



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