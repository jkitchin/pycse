import bottle
from beaker.middleware import SessionMiddleware
from cork import Cork
import logging

logging.basicConfig(format='localhost - - [%(asctime)s] %(message)s', level=logging.DEBUG)
log = logging.getLogger(__name__)
bottle.debug(True)

# Use users.json and roles.json in the local example_conf directory
aaa = Cork('example_conf',
           email_sender='johnrkitchin@gmail.com',
           smtp_server='starttls://jkitchin:Nu3Quep4*@smtp.andrew.cmu.edu:587')

M = aaa.mailer

M.send_email('jkitchin@andrew.cmu.edu','test subject', 'testing')


## # Import smtplib for the actual sending function
## import smtplib

## # Import the email modules we'll need
## from email.mime.text import MIMEText

## # Open a plain text file for reading.  For this example, assume that
## # the text file contains only ASCII characters.

## msg = MIMEText('test')

## # me == the sender's email address
## # you == the recipient's email address
## msg['Subject'] = 'The contents of a string'
## msg['From'] = 'jkitchin@andrew.cmu.edu'
## msg['To'] = 'johnrkitchin@gmail.com'

## # Send the message via our own SMTP server, but don't include the
## # envelope header.
## s = smtplib.SMTP('smtp.andrew.cmu.edu', 587)
## s.starttls()
## s.login('jkitchin','Nu3Quep4*')
## s.sendmail('jkitchin@andrew.cmu.edu', ['johnrkitchin@gmail.com'], msg.as_string())
## s.quit()
