# Disaster Response Pipeline Project

The Project consists of 3 parts:

1. ETL pipeline:
	- loads data from disaster_messages.csv and the corresponding disaster_categories.csv
	- transforms data into the appropriate form
	- saves data in DisasterResponse database (a SQL database)

2. ML pipeline:
	- loads data from DisasterResponse database
	- splits data into train and test sets
	- initialize and train multioutput SGDClassifier using Gridsearch to find the best parameters
	- save model
	
3. Flask Web app:
	- load model
	- display the distribution of the data
	- classify custom messages according to the categories (1 or 0)


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
