import commands, os
env=Environment(ENV=os.environ, PDFLATEXFLAGS='-shell-escape')


# http://www.scons.org/doc/2.0.1/HTML/scons-user.html#chap-builders-commands
def build_tex(target, source, env): 
    print 'building tex file!!!!'
    status,output = commands.getstatusoutput('emacs --batch -l ~/Dropbox/.emacs.d/init.el --visit=pycse.org --funcall org-export-as-latex')
    print status, output



env.Command('pycse.tex', 'pycse.org', build_tex)

env.PDF('pycse.tex')
