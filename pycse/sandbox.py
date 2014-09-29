#!/usr/bin/env python
from cStringIO import StringIO
import os, sys

content = sys.stdin.read()
     
old_stdout = sys.stdout
old_stderr = sys.stderr
redirected_output = sys.stdout = StringIO()
redirected_error = sys.stderr = StringIO()

out, err, exc = None, None, None

    
exec(content)
  
out = redirected_output.getvalue()
err = redirected_error.getvalue()

sys.stdout = old_stdout
sys.stderr = old_stderr


s = '''sandbox:
---stdout-----------------------------------------------------------
{0}
'''.format(out)

if err:
    s += '''---stderr-----------------------------------------------------------
{0}
'''.format(err)

if exc:
    s += '''---Exception--------------------------------------------------------
{0}
'''.format(exc)

# print final result to stdout
print s
