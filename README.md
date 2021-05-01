# Build Disaster Response Pipelines with Figure Eight

### Project Motivation
This project is part of the Udacity Data Science Nanodegree program


Figure Eight, a company focused on creating datasets for AI applications, has crowdsourced the tagging and translation of messages to improve disaster relief efforts. In this project, we build a data pipeline to prepare message data from major natural disasters around the world. We build a machine learning pipeline to categorize emergency messages based on the needs communicated by the sender.

The project provides a web app where you can input a text emergency message and receive a classification in different emergency categories.  During natural disasters, a large number of emergency messages reach emergency services via social media or direct contact. Categorizing those messages via AI helps disaster response organizations to filter for the most relevant information and to allocate the messages to the relevant rescue teams.



### Project Descriptions
The project consists of three parts and the datasets:

***ETL Pipeline:*** `process_data.py` file with python code to create an ETL pipeline.
    
Build an ETL pipeline (Extract, Transform, Load) to retrieve emergency text messages and their classification from a given dataset. Clean the data and store it in an SQLite database.

***ML Pipeline:*** `train_classifier.py` file contains the python code to create an ML pipeline.

Divide the data set into a training and test set. Create a sklearn machine learning pipeline using NLTK (Natural Language Toolkit) using Hyperparameter optimization via Grid Search.  The ml model uses the AdaBoost algorithm (formulated by Yoav Freund and Robert Schapire) to predict the classification of text messages (multi-output classification).  

***Web App:***  
A web application enables the user to enter an emergency message, and then view the categories of the message in real time.


***Data***
The ml model trains on a dataset provided by Figure Eight that consists of 30,000 real-life emergency messages. The messages are classified into 36 labels.

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


### Licensing, Authors, and Acknowledgements

Thanks to Udacity for the starter code and FigureEight for providing the data set of 30,000 labelled emergency messages to be used in this project.

### Screenshots Web App

![grafik](https://user-images.githubusercontent.com/59873708/116595766-2da20e00-a913-11eb-987e-19daadda147a.png)

![grafik](https://user-images.githubusercontent.com/59873708/116595838-45799200-a913-11eb-9610-0aeff785e149.png)

![grafik](https://user-images.githubusercontent.com/59873708/116723934-8cc55880-a9cf-11eb-92a6-cd9673f96261.png)

