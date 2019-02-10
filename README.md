# Disaster Response Pipeline Project

### Files
1. app
    - Contains the run.py and HTML templates to run Flask Web App with final model results and graphs
2. data
    - Contains process_data.py to apply data pre-processing steps
    - disaster_messages.csv is the CSV of raw messages
    - disaster_categories.csv is the CSV of corresponding message categories
    - DisasterResponse.db is the database that contains the Messages table
3. models
    - train_classifier.py to build and train the model classifier
    - classifier.pkl is the pickle dump of the trained model

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/
