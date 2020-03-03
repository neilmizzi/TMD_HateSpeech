from flask import Flask, redirect, url_for, request, render_template
app = Flask(__name__)


"""
   CLEARS CACHE
"""
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


"""
   INDEX PAGE
"""
@app.route('/')
def search_form():
   return render_template('search_form.html')


"""
   RESULTS PAGE
   -  Gets data from form on index

   TODO
   -  Retrieve tweets from twint_handler.py
   -  Get prediction from NN/SVM
   -  Pass appropriate params to search_results.html
"""
@app.route('/search_results',methods = ['POST', 'GET'])
def search_results():
   if request.method == 'POST':
      #TODO Twint Data Retrieval
      #     Get Prediction
      #     Pass params
      user = request.form['user']
      return render_template("search_results.html", user=user)
   else:
      # TODO Replace with 404 webpage
      return render_template("search_results.html", user="NaN")


if __name__ == '__main__':
   app.run(debug=True)
