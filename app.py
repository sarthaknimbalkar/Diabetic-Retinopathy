import os
import re
import logging
import numpy as np
from PIL import Image
from flask import Flask, redirect, url_for, request, render_template, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, current_user, logout_user
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from gevent.pywsgi import WSGIServer
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.preprocessing import image
from models.model import build_model
from datetime import datetime
from tensorflow.keras.backend import clear_session



# Initialize the Flask application
app = Flask(__name__)
login_manager = LoginManager()
app.config['SECRET_KEY'] = 'any-secret-key-you-choose'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///login.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['DEBUG'] = True

# Initialize SQLAlchemy and Flask-Login
db = SQLAlchemy(app)
login_manager.init_app(app)
login_manager.login_view = 'signin'

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Initialize the model
model = build_model()

# Create uploads directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Password strength regex pattern
password_pattern = re.compile(
    r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?=.*[!@#$%^&*()-_=+{};:,<.>]).{8,}$'
)

# Define User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

class FormData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(255), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    severity = db.Column(db.String(10), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now)

@app.route('/submit-form', methods=['POST'])
@login_required
def submit_form():
    name = request.form.get('name')
    description = request.form.get('description')
    phone = request.form.get('phone')
    location = request.form.get('location')
    severity = request.form.get('severity')

    new_data = FormData(
        name=name,
        description=description,
        phone=phone,
        location=location,
        severity=severity
    )

    db.session.add(new_data)
    db.session.commit()

    return jsonify(success=True, message="Form data saved successfully.")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def model_predict(img, model):
    clear_session()  # Clear the session to ensure no previous state is carried over
    img = img.resize((224, 224), Image.LANCZOS)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    img.close()
    return preds


def decode_predictions(preds):
    severity_levels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    predicted_class = np.argmax(preds, axis=1)[0]
    predicted_label = severity_levels[predicted_class]
    return predicted_label

@app.route('/', methods=['GET'])
def index():
    return render_template('login.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    logging.info("Received request at /predict")
    
    if 'image' not in request.files:
        logging.error("No image part in the request")
        return jsonify(success=False, error="No image part"), 400
    
    file = request.files['image']
    if file.filename == '':
        logging.error("No selected file")
        return jsonify(success=False, error="No selected file"), 400
    
    if file:
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            logging.info(f"File saved at {file_path}")

            img = Image.open(file_path)
            preds = model_predict(img, model)
            logging.debug(f"Predictions: {preds}")

            predicted_label = decode_predictions(preds)
            os.remove(file_path)  # Optional: remove the file after prediction
            logging.info(f"Prediction: {predicted_label}")
            return jsonify(success=True, prediction=predicted_label)
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return jsonify(success=False, error=str(e)), 500
        finally:
            if 'img' in locals():
                img.close()  # Ensure image file is closed in case of exceptions
    
    return jsonify(success=False, error="File upload failed"), 500


@app.route('/signup', methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        reg_email = request.form.get('EMAIL')
        reg_name = request.form.get('NAME')
        reg_password = request.form.get('PASSWORD')

        if not reg_password:
            flash("Password is required.")
            return render_template('login.html')

        if not re.match(password_pattern, reg_password):
            flash("Password must contain at least 8 characters, including one uppercase letter, one lowercase letter, one digit, and one special character.")
            return render_template('login.html')

        hashed_password = generate_password_hash(reg_password, method="pbkdf2:sha256", salt_length=8)
        existing_user = User.query.filter_by(email=reg_email).first()

        if existing_user:
            flash("You've already signed up with this email.")
            return render_template('login.html')

        new_user = User(name=reg_name, email=reg_email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('after_login'))

    return render_template('login.html')

@app.route('/signin', methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        email = request.form.get('EMAIL')
        password = request.form.get('PASSWORD')
        user = User.query.filter_by(email=email).first()

        if not user:
            flash("That email does not exist, please try again.")
            return redirect(url_for('signin'))
        elif not check_password_hash(user.password, password):
            flash('Password incorrect, please try again.')
            return redirect(url_for('signin'))
        else:
            login_user(user)
            return redirect(url_for('after_login'))

    return render_template('login.html')

@app.route('/afterlogin', methods=['GET'])
@login_required
def after_login():
    return render_template("upload.html")

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    http_server = WSGIServer(('0.0.0.0', 5002), app)
    http_server.serve_forever()
