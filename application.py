from flask import Flask, redirect, url_for, request, render_template
import pandas as pd
app = Flask(__name__)

def get_urls():
    df = pd.read_csv('Arkivji/trump.csv', sep=',')
    urls = df.loc[:, "link"]
    tweetlist = []
    for url in urls:
        url = url.split('/')
        tweetlist.append(url[-1])
    return tweetlist

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


@app.route('/')
def search_form():
   return render_template('search_form.html')


@app.route('/search_results',methods = ['POST', 'GET'])
def search_results():
   if request.method == 'POST':
      #TODO Data processing goes here
      user = request.form['user']
      tweetlist = get_urls()
      return render_template("search_results.html", user=user, tweetlist=tweetlist[:5])


if __name__ == '__main__':
   app.run(debug=True)
