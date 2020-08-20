from flask import Flask, request, redirect, session, send_from_directory, Request
from functools import wraps

import json

class LoginRequired:
    def __init__(self, app, usersjson='nerdlogin.json'):
        url_map = app.url_map
        self.hooked_endpoints = [r.rule for r in url_map.iter_rules()]
        print(self.hooked_endpoints)
        self.app = app
        self.app.config["SECRET_KEY"] = "a very hard to guess secret key"
        self._init_all()
        
        self.usersjson = usersjson
        if usersjson:
            try:
                with open(usersjson) as inp:
                    self.users = json.load(inp)
            except Exception:
                print(f'Failed to load {usersjson}')
                self.users = {}
        else:
            self.users = {}
        
    def hook(self):
        print('endpoint: %s, url: %s, path: %s' % (
        request.endpoint,
        request.url,
        request.path))
        if request.path in self.hooked_endpoints:
            loggedin = self._is_user_logged_in(session)
            if not loggedin:
                return redirect('/login')
            
    
    def _init_all(self):
        self.app.before_request(self.hook)
        self.app.add_url_rule('/login', 'login', self.login, methods=['GET', 'POST'])
        self.app.add_url_rule('/register', 'register', self.register, methods=['POST'])
        self.app.add_url_rule('/logout', 'logout', self.logout, methods=['GET'])
        
    
    def login(self):
        if request.method == 'GET':
            return send_from_directory('html_templates/', 'usermanagement.html')

        elif request.method == 'POST':
            data = json.loads(request.data)
            validation = self.validate_user(data)
            if validation == 'Valid User':
                session['username'] = data['username']
                return {
                    'redirect': '/'
                }
            else:
                return {
                    'error': validation
                }

    def logout(self):
        session.pop('username', None)
        return redirect('/login')
    
    def register(self):
        data = json.loads(request.data)
        username = data.get('username', None)
        password = data.get('password', None)
        print(data)
        if username in self.users:
            return {
                'error': 'User already exists'
            }
        
        else:
            self.users[username] = {
                'password': password
            }
            session['username'] = username
            with open(self.usersjson, 'w') as out:
                json.dump(self.users, out)
            
            return {
                    'redirect': '/'
            }
            
            
    def validate_user(self, userd):
        user = self.users.get(userd['username'], None)
        if user:
            password = user['password']
            if password == userd['password']:
                return 'Valid User'
            else:
                return 'Wrong Password'
        else:
            return 'Invalid Username'


    def _is_user_logged_in(self, session):
        if 'username' not in session:
            return False
        return True
