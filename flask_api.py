from flask import Flask
app = Flask(__name__)

# This code is obtained from following Flask TutorialPoint guide
# URL: https://www.tutorialspoint.com/flask/index.htm
# This script lists a number of applicable examples on how Flask works

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



# MAIN
if __name__ == '__main__':
    app.debug = True
    app.run()
    app.run(debug=True)
