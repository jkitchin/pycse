#!python

'''
publish  *.py

Convert a python script to a published format that has pretty-printed code, captured output, including figures.


This script is a wrapper around pyreport:
http://gael-varoquaux.info/computers/pyreport/

install that package like this:
pip install --upgrade https://github.com/joblib/pyreport/archive/master.zip

That package only wraps pylab, and I use matplotlib.pyplot a lot. This script wraps that with modified functions that capture the output, but still leverages the pyreport code.
'''
import cStringIO, os, sys, traceback
#import random, string

# Adapted from pyreport to patch the matplotlib.pyplot.show function
from pyreport import main
    
import matplotlib.pyplot
matplotlib.pyplot.show = main.myshow

original_savefig = matplotlib.pyplot.savefig

def mysave(*args, **kwargs):
    self = main.myshow
    
    figure_name = '%s%d.%s' % ( self.basename,
                                len(self.figure_list),
                                self.figure_extension )
    self.figure_list += (figure_name, )
    print "Here goes figure %s" % figure_name
    import pylab
    original_savefig(figure_name)

matplotlib.pyplot.savefig = mysave

    
##################################################################
##################################################################
import argparse
from uuid import getnode as get_mac
import re, socket
import datetime
from pyreport import pyreport, options

data = {}
## userid = os.environ.get('USER','no user found')
## date_submitted = datetime.datetime.now()

## # mac address of submitting computer
## mac = get_mac()
## try:
##     hostname, aliases, ipaddr =  socket.gethostbyaddr(socket.gethostbyname(socket.gethostname()))
## except:
##     hostname, aliases, ipaddr = None, None, None

## if ipaddr:
##     ipaddr = ipaddr[0] # it is usually a list. I think it is ok to take the first element.
## data = {'mac':mac,
##         'hostname':hostname,
##         'ipaddr':ipaddr,
##         'userid':userid,
##         'date_submitted':date_submitted}

PROPERTIES = ['COURSE',
              'ASSIGNMENT',
              'ANDREWID',              
              'NAME']

##################################################################

parser = argparse.ArgumentParser(description='submit your python script and output in tex, pdf or org-mode archive file')

parser.add_argument('files', nargs='*',                    
                    help='scripts to submit')

args = parser.parse_args()
    
for INPUT in args.files:
    # check for compliance of data
    with open(INPUT) as f:
        text = f.read()

        for prop in PROPERTIES:
            regexp = '#\+{0}:(.*)'.format(prop)
            m = re.search(regexp, text)
            if m:
                data[prop] = m.group(1).strip()
            else:
                raise Exception('''You are missing #+{0}: in your file. please add it and try again.'''.format(prop))

    BASENAME = '{ANDREWID}-{COURSE}-{ASSIGNMENT}'.format(**data)
    
    opts, args = options.parse_options(['-o',
                                        '{0}.pdf'.format(BASENAME),
                                        #'-v',
                                        #'-t','pdf',
                                        '-l', #allow LaTeX literal comment lines starting with "#$"
                                        '-e' #allow LaTeX math mode escape in code wih dollar signs
                                        ])        
    opts.update({'infilename':INPUT})

    default_options, _not_used = options.option_parser.parse_args(args =[])
    default_options.figure_type = 'png'
    
    pyreport.main(open(INPUT), overrides=opts)


    
    




