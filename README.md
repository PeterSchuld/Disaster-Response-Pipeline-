# Build Disaster Response Pipelines with Figure Eight

### Project Motivation
This project is part of the Udacity Data Science Nanodegree program


Figure Eight, a company focused on creating datasets for AI applications, has crowdsourced the tagging and translation of messages to improve disaster relief efforts. In this project, we will build a data pipeline to prepare message data from major natural disasters around the world. We will build a machine learning pipeline to categorize emergency messages based on the needs communicated by the sender.

The project provides a web app where you can input a text message and receive a classification in different emergency categories.  During natural disasters, a large number of emergency messages reach emergency services via social media or direct contact. Categorizing those messages via AI helps disaster response organizations to filter for the most relevant information and to allocate the messages to the relevant rescue teams. 


### Installation:
You need python3 and the following libraries installed to run the project: 
   
    - pandas
    - re
    - sys
    - json
    - sklearn
    - nltk
    - sqlalchemy
    - sqlite3
    - pickle
    - Flask
    - plotly


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Screenshots Web App

![grafik](https://user-images.githubusercontent.com/59873708/116595766-2da20e00-a913-11eb-987e-19daadda147a.png)

![grafik](https://user-images.githubusercontent.com/59873708/116595838-45799200-a913-11eb-9610-0aeff785e149.png)


