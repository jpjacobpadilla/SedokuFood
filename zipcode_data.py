import re
from flask import  jsonify


def zip_lookup(request, db):
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