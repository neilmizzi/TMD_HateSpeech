from flask import Flask, redirect, url_for, request, render_template
app = Flask(__name__)

@app.route('/')
def search_form():
   return render_template('search_form.html')

@app.route('/search_results',methods = ['POST', 'GET'])
def search_results():
   if request.method == 'POST':
      #TODO Data goes here
      user = request.form['user']
      return render_template("search_results.html", user=user)

if __name__ == '__main__':
   app.run(debug = True)