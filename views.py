from difflib import restore
from tracemalloc import start
from flask import Flask, render_template, url_for, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from project_secrets import user_db, password_db, host_db, name_db
import dish_recommendor
import zipcode_data


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


@app.route('/dishfinder', methods=['GET'])
def dishfinder():
    return render_template('dishfinder.html')


@app.route('/explorezipcodeprocess', methods=['POST'])
def explorezipcodeprocess():
    json_resp = zipcode_data.zip_lookup(request, db)
    return json_resp


@app.route('/dishfinderalgo', methods=['POST'])
def dishfinderalgo():
    json_resp =  dish_recommendor.dish_rec_main(request, db)
    return json_resp


if __name__ == '__main__':
    app.run(debug=True)
