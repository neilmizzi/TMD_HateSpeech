from flask import Flask, redirect, url_for, request, render_template, abort
import pandas as pd
from classifierrun import lstm_predictions, restructure_results
from twint_handler import TwintHandler
from get_graph import get_chart
app = Flask(__name__)


# CLEAR CACHE
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


@app.errorhandler(404)
def page_not_found(error):
   return render_template('404.html', title = '404'), 404


@app.route('/')
def search_form():
   return render_template('search_form.html')


@app.route('/search_results',methods = ['POST', 'GET'])
def search_results():
   if request.method == 'POST':

      # Get form input
      user = request.form['user']
      limit = request.form['limit']
      date_lower = request.form['date_lower']
      date_upper = request.form['date_upper']
      df = pd.DataFrame()

      # Get tweets from Twint given parameters
      twint_handler = TwintHandler()
      try:
         df = twint_handler.search_user(user, limit, date_lower, date_upper)
         urls = df.loc[:, "link"]
         tweet_text = df.loc[:, "tweet"]

      except Exception:
         abort(404)

      # TODO add flask visualization (bokeh pycharts etc)
      classification = lstm_predictions()
      results = restructure_results(classification)

      tweet_list = zip(
         [i for i in range(len(tweet_text))],
         tweet_text,
         urls,
         results
         )

      script, div = get_chart(results)

      return render_template(
         "search_results.html", 
         user = user,  
         tweet_list = tweet_list,
         script = script,
         div = div
      )

   abort(404)


if __name__ == '__main__':
   app.run(debug=True)