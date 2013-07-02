# -*- python -*-
'''
scons -c  # clean up
scons pycse.tex  # create tex file
scons pycse.pdf  # build pdf file

'''

import commands, os
env=Environment(ENV=os.environ, PDFLATEXFLAGS='-shell-escape')

EMACS = 'emacs'
EMACS_OPTIONS = [
                 '-l ~/Dropbox/.emacs.d/init.el',
                 '-l ~/Dropbox/pycse/pycse.el']


# http://www.scons.org/doc/2.0.1/HTML/scons-user.html#chap-builders-commands
def build_tex(target, source, env): 
    print 'building tex file!!!!'
    cmd = (EMACS + ' ' 
           + ' '.join(EMACS_OPTIONS)
           + ' --visit=pycse.org --funcall=org-latex-export-to-latex')
   
    status, output = commands.getstatusoutput(cmd)
    print status, output

def build_html(target, source, env): 
    print 'building html file!!!!'
    cmd = (EMACS + ' ' 
           + ' '.join(EMACS_OPTIONS)
           + ' --visit=pycse.org')
   
    status, output = commands.getstatusoutput(cmd)
    print output
    
# Build the tex file
env.Command('pycse.tex', 'pycse.org', build_tex)

# Build the html file
env.Command('pycse.html', 'pycse.org', build_html)

# build the pdf file. this automatically runs bibtex, makeindex and pdflatex as many times as needed. I am not sure how to make this depend on the images too. This is probably unneeded, since I only change an image by changing the org-file.
pdf_src = ['pycse.tex']
env.PDF(target='pycse.pdf', source=pdf_src)





