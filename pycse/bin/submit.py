#!/usr/bin/env python
'''
Script to submit a file to the homework server

files must be text based, and they must contain these lines:

#+COURSE: 06-625
#+ASSIGNMENT: 1a
#+EMAIL: jkitchin@andrew.cmu.edu
'''

import requests
from uuid import getnode as get_mac
import socket
import os
import uuid
import datetime
import sys
import getpass
import re

url = 'http://localhost:8080/upload'

# mac address of submitting computer
mac = get_mac()

try:
    hostname, aliases, ipaddr =  socket.gethostbyaddr(socket.gethostbyname(socket.gethostname()))
except:
    hostname, aliases, ipaddr = None, None, None
    
userid = os.environ.get('USER','no user found')
date_submitted = datetime.datetime.now()

#password = getpass.getpass('Password: ')

# These should be in #+PROP: lines in the file. It is an error if they do not exist.
PROPERTIES = ['COURSE',
              'ASSIGNMENT',
              'ANDREWID',
              'EMAIL',
              'NAME']

data = {'mac':mac,
        'hostname':hostname,
        'ipaddr':ipaddr,
        'userid':userid,
        'date_submitted':date_submitted}

for fname in sys.argv[1:]:

    with open(fname) as f:
        text = f.read()
    
        for prop in PROPERTIES:
            regexp = '#\+{0}:(.*)'.format(prop)
            m = re.search(regexp, text)
            if m:
                data[prop] = m.group(1).strip()
            else:
                raise Exception('''You are missing #+{0} in your file. please add it and try again.'''.format(prop))
        
    files = {'file': open(fname, 'rb')}
    r = requests.post(url, files=files, data=data)

    print r.status_code
    print r.text


