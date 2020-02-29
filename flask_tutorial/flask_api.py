from flask import Flask, redirect, url_for
app = Flask(__name__)

# This code is obtained from following Flask TutorialPoint guide
# URL: https://www.tutorialspoint.com/flask/index.htm
# This script lists a number of applicable examples on how Flask works and can be used in our app

"""INDEX PAGE"""
@app.route('/')
def index():
    return 'Hello there, I see you wish to lurk Tweets.'    # Returns HTML


"""NORMAL PAGE REDIRECT, NO INPUTS"""
@app.route('/hello')
def hello_world():
   return '<h1>something something test</h1>'   # returns HTML, can be parsed accordingly


"""EXAMPLE PASSING PARAMETER (We can use this to pass twitter user info)"""
@app.route('/hello/<name>')
def hello_name(name):
    if name == 'Neil':
        return '<h1>You\'re a legend.</h1>'
    if name == 'Britt':
        return '<h1>Banana hater!</h1>'
    return 'Hello <b>%s</b>!' % name


"""Passing integer value"""
@app.route('/blog/<int:postID>')
def blog_post(postID):
    return 'Blog Numer %d' % postID



"""REDIRECT EXAMPLES"""
@app.route('/admin')
def hello_admin():
   return 'Hello Admin'


@app.route('/guest/<guest>')
def hello_guest(guest):
   return 'Hello %s as Guest' % guest


@app.route('/user/<name>')
def hello_user(name):
   if name =='admin':
      return redirect(url_for('hello_admin'))
   else:
      return redirect(url_for('hello_guest',guest = name))


# MAIN
if __name__ == '__main__':
    app.debug = True
    app.run()
    app.run(debug=True)
