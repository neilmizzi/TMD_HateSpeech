# LSTM for hatespeech recognition
### (Guido Ansem, Thomas de Gier, Borach Jansema, Britt van Leeuwen, Neil Mizzi - The Vrije Universiteit Amsterdam)

This project contains code for the development and optimization of the hyperparameters of a LSTM for hatespeech recognition in tweets. The data used for training and testing is a combination of different datasets gathered from Kaggle. A different dataset of tweets scraped from multiple users was annotated and used for evaluation. The optimal LSTM framework was then used to run as backand on a web application for hatespeech recognition. 

## Prerequisites
This code requires the installation of the following Python packages:
-   Keras
-   Tensorflow
-   Flask
-   Pandas
-   Twint

# Running

## Classification
To start the optimization make sure that the data set is saved in the same directory as the code. If not adjust the path in "loading_processing_data.py" line 24 to the file location. Adjust the constant ITERATIONS in the tune_lstm.py file to the desired amount of iterations, then simply run the file.

## Application
Running "application.py" will start a local web server, and the URL to access the website itself will be indicated in the log.

Going to the index page, you will be presented with the input form. The only required field is the username to scrape. The rest are optional and may be left out, but offer better limiting options to include or eliminate tweets of a certain age, or limit the total number of tweets to return.

The results include a list of tweets, alongside the predicted classification of the LSTM. For better context, we include the actual text, and also a link to the tweet to see the replies and/or other information on Twitter itself.

If the username does not exist, or it does exist but they have not posted any tweets, then you will be redirected to the 404 page.


# Known Issues
-   Due to an issue with Twint on Windows, we are unable to load the scraped data directly as a Pandas Dataframe on memory. For now, we load a new .csv file every time and return those results as a workaround.
