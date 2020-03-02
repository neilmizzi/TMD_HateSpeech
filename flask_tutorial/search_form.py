from flask import Flask, redirect, url_for, request, render_template
app = Flask(__name__)

@app.route('/search_results/<user>')
def search_results(user):
   return render_template('search_results.html', user=user)

@app.route('/search_form',methods = ['POST', 'GET'])
def search_form():
   if request.method == 'POST':
      user = request.form['user']
      return redirect(url_for('search_results', user=user))
   else:
      user = request.args.get('user')
      return redirect(url_for('search_results',user=user))

if __name__ == '__main__':
   app.run(debug = True)