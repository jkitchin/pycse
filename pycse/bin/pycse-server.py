#!/usr/bin/env python

'''
Server prototype for turning in homeworks

The server will have a list of userids that are allowed to submit. A client command will check the following:

1. user is allowed to submit, ideally with authentication
2. assignment is expected and before the due date.

the client will then upload the assignment to the correct place. 

The server will have to know the users allowed to submit, what assignments can be submitted, and their due dates.

* TODO

- [x] Make user-defined variables for where to upload files
- [x] Move COURES, USERS, ASSIGNMENTS to text files that can be used in emacs
- [] add logging, especially for failures
- [] add git version control so that I can see how files have been changed.
'''
import commands, datetime, os
from bottle import route, run, template, get, post, request

HOST = 'localhost'
PORT = 8080

with open('COURSES') as f:
    COURSES = []
    for line in f:
        if line.startswith('#'): continue
        if line == '': continue
        COURSES.append(line.strip())

with open('USERS') as f:
    USERS = {}
    for line in f:
        if line.startswith('#'): continue
        if line == '': continue
        
        user, passwd = line.split()
        USERS[user] = passwd.strip()

with open('ASSIGNMENTS') as f:
    ASSIGNMENTS = {}
    for line in f:
        if line.startswith('#'): continue
        if line == '': continue
        assng, duedate = line.split('\t')
        ASSIGNMENTS[assng] = duedate.strip()

def overdue(assignment, date_submitted):
    'return whether an assignment is overdue'
    due_date = ASSIGNMENTS[assignment]
    # compare datetimes
    due = datetime.datetime.strptime(due_date,'%Y-%m-%d %H:%M:%S')
    submitted = datetime.datetime.strptime(date_submitted,'%Y-%m-%d %H:%M:%S.%f')
    return submitted > due

def submission_ok(fname, assignment, date_submitted):

    # if there is no file, and it is not overdue, then submittal is ok.
    if (not os.path.exists(fname)
        and not overdue(assignment, date_submitted)):
        print ' no file found and not overdue.'
        return True

    # a file does exist, so we check for a resubmission property
    with open(fname) as f:
        for line in f:
            if line.startswith('#+RESUBMIT:'):
                new_deadline = line[11:].strip()
                due = datetime.datetime.strptime(new_deadline,'%Y-%m-%d %H:%M:%S')
                submitted = datetime.datetime.strptime(date_submitted,'%Y-%m-%d %H:%M:%S.%f')
                print 'found resubmit: checking'
                return submitted < due
        print ' ok, no resubmit found, just check for due date'
        return not overdue(assignment, date_submitted)

successful_upload_template = '''#+TITLE: {{course}}/{{assignment}}
#+COURSE: {{course}}
#+ASSIGNMENT: {{assignment}}
#+NAME: {{name}}
#+EMAIL: {{email}}
#+ANDREWID: {{userid}}
#+DATE_SUBMITTED: {{date_submitted}}
#+IPADDR: {{ipaddr}}
#+HOSTNAME: {{hostname}}
#+MACADDR: {{mac}}

{{fs.filename}}

* {{assignment}}

%if fs.filename.endswith(".py"):
#+BEGIN_SRC python
%end
{{fs.value}}
%if fs.filename.endswith(".py"):
#+END_SRC
%end

* FEEDBACK
'''

unsuccessful_upload_template = '''Upload of {{fs.filename}} was not successful.
Please send this information to Professor Kitchin:

course:         {{course}}  
assignment:     {{assignment}}
andrew id:      {{andrewid}}

date submitted: {{date_submitted}}
due date:       {{due_date}}

userid:         {{userid}}
mac:            {{mac}}
ipaddr:         {{ipaddr}}
filename:       {{fs.filename}}
text:
{{fs.value}}
'''

@route('/')
def index():
    print ASSIGNMENTS
    return template('''
Welcome to 06-625!

<a href="/syllabus">Syllabus</a><br>
<br><br>

<h2> Assignments and due dates </h2>
<table border="1">
%for assgn in ASSIGNMENTS:
<tr>
<td>{{assgn}}</td><td>{{ASSIGNMENTS[assgn]}}</td>
</tr>
%end
</table>''', {'ASSIGNMENTS':ASSIGNMENTS})


@route('/syllabus')
def syllabus():
    "show syllabus"
    with open('syllabus.org') as f:
        text = f.read()
    return template('''<pre>{{text}}</pre>''', text=text)

@post('/upload')
def upload():
    '''upload an assignment. some validation is performed to make sure
    the right metadata exists, and that the file is an expected one'''
    course = request.forms.get('COURSE')
    andrewid = request.forms.get('ANDREWID')
    name = request.forms.get('NAME')
    email = request.forms.get('EMAIL')
    assignment = request.forms.get('ASSIGNMENT')
    date_submitted = request.forms.get('date_submitted')

    mac = request.forms.get('mac')
    ipaddr = request.forms.get('ipaddr')
    userid = request.forms.get('userid')
    date_submitted = request.forms.get('date_submitted')
    hostname = request.forms.get('hostname')

    fs = request.files.get('file', None)
   
    # here is the minor validation
    if not (course in COURSES
        and andrewid in USERS
        and assignment in ASSIGNMENTS):
        due_date = ASSIGNMENTS[assignment]
        return  template(unsuccessful_upload_template, **locals())

    # this is the directory where the file should be uploaded
    fdir = '{course}/{assignment}'.format(**locals())
    fname = os.path.join(fdir, '{andrewid}.org'.format(**locals()))

    if submission_ok(fname, assignment, date_submitted):

        if not os.path.exists(fdir):
            os.makedirs(fdir)

        with open(fname, 'wb') as f:
            f.write(template(successful_upload_template, **locals()))
            commands.getstatusoutput('git add {0}'.format(fname))
            commands.getstatusoutput('git commit {0} -m "upload"'.format(fname))
            
        return  template('Upload of {{fs.filename}} was successful', **locals())
    else:
        due_date = ASSIGNMENTS[assignment]
        return  template(unsuccessful_upload_template, **locals())


# authenticate
@route('/assignment/<course>/<assignment>/<andrewid>')
def get_assignment(course,  assignment, andrewid):
    pf = os.path.join(course, assignment, andrewid + '.org')
    if os.path.exists(pf):
        with open(pf) as f:
            text = f.read()
            return template('<pre>{{text}}</pre>', text=text)
    else:
        return pf + ' not found'

# TODO get final grade
@route('/student-grades/<course>/<andrewid>')
def get_student_grades(course, andrewid):
    'walk assignments and collect grades for a student'
    
    student = []
    for assignment in ASSIGNMENTS:
        pf = os.path.join(course, assignment, andrewid + '.org')
        url = '/assignment/{course}/{assignment}/{andrewid}'.format(**locals())
        student.append((assignment, get_grade(pf), url))
        
    return template('''
    <h1>grade report for {{andrewid}}
<table border="1">
<tr>
%for assgn,grd,url in student:
<td>{{assgn}}</td> <td>{{grd}}</td><td> <a href="{{url}}">link</a></td>
</tr>
%end
</table>''', **locals())
                       


def get_grade(pf):
    'get grade from a file pf if it exists'
    if not os.path.exists(pf):
        return 'No file'

    with open(pf) as f:
        for line in f:
            if line.startswith('#+GRADE:'):
                return line[8:]

    return 'ungraded'
            
    

if __name__ == '__main__':
    run(host=HOST, port=PORT, debug=True)
