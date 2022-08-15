# Disaster Response Model

This model is designed to classify messages sent in disaster situations.

It is trained on real world events sent during disaster events and can then take a new message as an input and predict the category of message.

During such situations, first responders can be flooded with a variety of different messages. It is tremendously valuable for communities to have the ability to sort through and organize these messages to help deliver approriate responses.

## Files in this Repository

***app***
- master.html # main page of web app
- go.html # classification result page of web app
- run.py # Flask file that runs app

***data***
- disaster_categories.csv # data to process
- disaster_messages.csv # data to process
- process_data.py #python notebook that processes the above csv files

***models***
- train_classifier.py #python notebook that creates and trains the classification model

***other***
- README.md

## Instructions

Before running the code, make sure you have installed plotly.express, WordCLoud and ligthgbm.

***1. Clean data***

Run the following code:

*python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db*

This runs the "process_data" notebook, takes the "disaster_messages" and 'disaster_categories" files, cleans the data and assigns them to "DisasterResponse".

***2. Train model***

Run the following code:

*python train_classifier.py DisasterResponse.db classifier.pkl*

This takes the data from "DisasterResponse" to train the model store in the "classifier" file.

***3. Run the app***

Run the following code:

*python run.py*

***4. Use the app***

In your web browser, go to the app. Then, input a message, click "Classify Message" and then you can see the result.
