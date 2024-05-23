### README

# Diabetic Retinopathy Detection using Artificial Intelligence

## Overview
This project aims to detect Diabetic Retinopathy (DR) using a Convolutional Neural Network (CNN) model built with TensorFlow and Keras, integrated into a Flask web application. The application allows users to sign up, log in, and submit retinal images for DR severity prediction.

## Features
- **User Authentication:** Users can sign up and log in using Flask-Login.
- **Form Submission:** Authenticated users can submit a form with details about their condition.
- **Image Upload and Prediction:** Users can upload retinal images, which are processed and analyzed by pre-trained DenseNet121, DenseNet201, and InceptionV3 models to predict the severity of DR.
- **Model:** The model predicts the severity level of DR, categorized as "No DR", "Mild", "Moderate", "Severe", or "Proliferative DR".

## Installation

### Prerequisites
- Python 3.7+
- Virtualenv (recommended)

### Dependencies
Install the necessary packages using the requirements.txt file:
```sh
pip install -r requirements.txt
```

### Database Setup
Initialize the SQLite database:
```sh
flask db init
flask db migrate
flask db upgrade
```

## Configuration
Configure your Flask application by setting the following environment variables:
- `SECRET_KEY`: A secret key for session management.
- `SQLALCHEMY_DATABASE_URI`: The database URI, default is 'sqlite:///login.db'.

## Running the Application
Start the Flask development server:
```sh
flask run
```
Alternatively, use Gevent WSGIServer for production:
```sh
python app.py
```

## Usage

### User Authentication
- **Signup:** Navigate to `/signup` to create a new user account.
- **Signin:** Navigate to `/signin` to log in with an existing account.

### Form Submission
- Authenticated users can submit their details via `/submit-form`.

### Image Prediction
- Navigate to `/afterlogin` to upload a retinal image for prediction.
- The prediction result will be displayed on the screen.

## Project Structure
```
/uploads          - Directory for storing uploaded images
/templates        - HTML templates for the Flask app
/models           - Directory for ML model definition
/static           - Static files (CSS, JS, images)
/instance         - SQLite database (generated after first run)
app.py            - Main Flask application
requirements.txt  - Python dependencies
```

## Model Details
The models used are DenseNet121, DenseNet201, and InceptionV3, which include:
- **Dense Blocks:** Multiple dense blocks for feature reuse.
- **Transition Layers:** Layers to control feature map growth.
- **Bottleneck Layers:** Layers to improve computational efficiency.
- **Global Average Pooling:** For aggregating spatial information.

### Data Preprocessing
- Images are resized to 224x224 pixels.
- Images are converted to arrays and preprocessed for DenseNet121.

### Performance Metrics
- **Accuracy**
- **Sensitivity**
- **Specificity**
- **Precision**
- **F1 Score**
- **AUC-ROC**
- **Confusion Matrix**

## Contributions
- **Sarthak**
- **Harshith**
- **Shashwat Shivam**

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Detailed literature and methodologies on detecting diabetic retinopathy using various CNN architectures and preprocessing techniques.
- Contributions from Kaggle notebooks and research papers on diabetic retinopathy detection.

For further details, refer to the `Details.pptx` presentation file.

---
