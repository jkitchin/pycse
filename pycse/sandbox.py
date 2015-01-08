#!/usr/bin/env python
'''Module that runs as a script to redirect stderr to stdout.
Output is designed for org-mode.'''

from cStringIO import StringIO
import sys

content = sys.stdin.read()

old_stdout = sys.stdout
old_stderr = sys.stderr
redirected_output = sys.stdout = StringIO()
redirected_error = sys.stderr = StringIO()

out, err, exc = None, None, None

# execute the code
exec(content)

out = redirected_output.getvalue()
err = redirected_error.getvalue()

sys.stdout = old_stdout
sys.stderr = old_stderr


s = '''{0}'''.format(out)

if err:
    s += '''
#+STDERR:
{0}
'''.format(err)

if exc:
    s += '''
#+EXCEPTIONS:
{0}
'''.format(exc)

# print final result to stdout
print(s)
