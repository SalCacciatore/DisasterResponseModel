# DisasterResponseModel

This model is designed to classify messages sent in disaster situations.

It is trained on real world events sent during disaster events and can then take a new message as an input and predict the category of message.



## Instructions

Before running the code, make sure you have install plotly.express and ligthgbm.

1. Clean data

Run the following code:

python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

This runs the "process_data" notebook, takes the "disaster_messages" and 'disaster_categories" files, cleans the data and assigns them to "DisasterResponse".

2. Train model

Run the following code:

python train_classifier.py DisasterResponse.db classifier.pkl

This takes the data from "DisasterResponse" to train the model store in the "classifier" file.

3. Run the app

Run the following code:

python run.py
