from flask import Flask, render_template, request, session, url_for,redirect
from flask_sqlalchemy import SQLAlchemy
import os
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
from datetime import datetime
from transformers import pipeline
from model import PositiveParaphraser
import mysql.connector



app = Flask(__name__)
auth = HTTPBasicAuth()
paraphraser = PositiveParaphraser()

paraphraser.load_model()

# Configuration for the database (replace with your database URI)
#app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:root@localhost/sentiment_logs'
#app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # to suppress a warning
#db = SQLAlchemy(app)
db_connection = mysql.connector.connect(
    host='localhost',
    port="3306",
    user='root',
    password='root',
    database='sentiment_logs'
)



# Define a simple user for basic authentication
users = {"username": generate_password_hash("password")}
app.secret_key = os.urandom(24)





@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('predict'))
    return render_template('index.html')

# Login endpoint
@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    if username == 'user1' and password == 'password1':
        # Authentication successful, set session and redirect to predict page
        session['username'] = username
        return redirect(url_for('predict_sentiment'))
    else:
        # Authentication failed, redirect back to home page
        return render_template('index.html', error='Invalid username or password')


# API endpoint for sentiment prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict_sentiment():
    if request.method == 'POST':
        try:
            user_sentence = request.form['text']

            # Assuming paraphraser is defined elsewhere
            paraphrased_sentence = paraphraser.paraphrase_positive(user_sentence)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print(f"User Sentence: {user_sentence}")
            print(f"Paraphrased Sentence: {paraphrased_sentence}")

            cursor = db_connection.cursor()
            query = "INSERT INTO prediction_log (timestamp, user_sentence, paraphrased_sentence) VALUES (%s, %s, %s)"
            values = (timestamp, user_sentence, paraphrased_sentence)
            cursor.execute(query, values)
            db_connection.commit()

            return render_template('result.html', user_sentence=user_sentence, paraphrased_sentence=paraphrased_sentence)

        except Exception as e:
            return render_template('result.html', prediction=f'Error: {str(e)}')

    return render_template('predict.html')




# Logout endpoint
@app.route('/logout',methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

