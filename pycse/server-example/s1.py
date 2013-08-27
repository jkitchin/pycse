from bottle import route, run
@route('/')
@route('/hello')
def hello():
    return 'Hello World!<a href="/upload">upload</a>'

@route('/')

@get('/login') # or @route('/login')
def login_form():
    return '''<form method="POST" action="/login">
                <input name="name"     type="text" />
                <input name="password" type="password" />
                <input type="submit" />
              </form>'''

def check_login(name, password):
    return True

@post('/login') # or @route('/login', method='POST')
def login_submit():
    name     = request.forms.get('name')
    password = request.forms.get('password')
    if check_login(name, password):
        return "<p>Your login was correct</p>"
    else:
        return "<p>Login failed</p>"
    
run(host='localhost', port=8080, debug=True)
