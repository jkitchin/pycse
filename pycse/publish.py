#!python

'''
publish  *.py

Convert a python script to a published format that has pretty-printed code, captured output, including figures.


This script is a wrapper around pyreport:
http://gael-varoquaux.info/computers/pyreport/

install that package like this:
pip install --upgrade https://github.com/joblib/pyreport/archive/master.zip

That package only wraps pylab, and I use matplotlib.pyplot a lot. This script wraps that with modified functions that capture the output, but still leverages the pyreport code.


# This is some major monkey patching to get the following:
# 1. paper size, with default opening to fit width in window
# 2. insert user data into file.
# 3. attach python script to the file
'''
import os
from pyreport import main
from pyreport.main import rst2latex, protect, tex2pdf, safe_unlink, ReportCompiler
import re
from docutils import core as docCore
from docutils import io as docIO

VERSION = 3.0

class MyTexCompiler(ReportCompiler):
    global data
    empty_listing = re.compile(
            r"""\\begin\{lstlisting\}\{\}\s*\\end\{lstlisting\}""", re.DOTALL)
    
    inputBlocktpl = r"""
    
.. raw:: LaTeX

    {\inputBlocksize
    \lstset{escapebegin={\color{darkgreen}},backgroundcolor=\color{lightblue},fillcolor=\color{lightblue},numbers=left,name=pythoncode,firstnumber=%(linenumber)d,xleftmargin=0pt,fillcolor=\color{white},frame=single,fillcolor=\color{lightblue},rulecolor=\color{lightgrey},basicstyle=\ttfamily\inputBlocksize}
    \begin{lstlisting}{}
    %(textBlock)s
    \end{lstlisting}
    }
    
    
"""
    outputBlocktpl =  r"""
.. raw:: LaTeX

    \lstset{backgroundcolor=,numbers=none,name=answer,xleftmargin=3ex,frame=none}
    \begin{lstlisting}{}
    %s
    \end{lstlisting}
    
"""
    errorBlocktpl = r"""

.. raw:: LaTeX


    {\color{red}{\bfseries Error: }
    \begin{verbatim}%s\end{verbatim}}
    
"""
    figuretpl = r'''
    \end{lstlisting}
    \\centerline{\includegraphics[scale=0.5]{%s}}
    \\begin{lstlisting}{}'''
    
    def __init__(self, options):
        self.preamble = r"""
    \usepackage{{listings}}
    \usepackage{{color}}
    \usepackage{{graphicx}}
    \usepackage{{attachfile}}
    \usepackage{{amsmath}}
    \usepackage{{datetime}}
    % set some pdf metadata
    \pdfinfo{{
             /AndrewID ({andrewid})  
             /Assignment ({assignment})
             /Grade (ungraded)
             }}
    \definecolor{{darkgreen}}{{cmyk}}{{0.7, 0, 1, 0.5}}
    \definecolor{{darkblue}}{{cmyk}}{{1, 0.8, 0, 0}}
    \definecolor{{lightblue}}{{cmyk}}{{0.05,0,0,0.05}}
    \definecolor{{grey}}{{cmyk}}{{0.1,0.1,0.1,1}}
    \definecolor{{lightgrey}}{{cmyk}}{{0,0,0,0.5}}
    \definecolor{{purple}}{{cmyk}}{{0.8,1,0,0}}

\begin{{popupmenu}}{{myMenu}}
    \item{{title=A++}}
    \item{{title=A+}}
    \item{{title=A}}
    \item{{title=A-}}
    \item{{title=A/B}}
    \item{{title=B+}}
    \item{{title=B}}
    \item{{title=B-}}
    \item{{title=B/C}}
    \item{{title=C+}}
    \item{{title=C}}
    \item{{title=C-}}
    \item{{title=C/D}}
    \item{{title=D+}}
    \item{{title=D}}
    \item{{title=D-}}
    \item{{title=D/R}}
    \item{{title=R+}}
    \item{{title=R}}
    \item{{title=R-}}
    \item{{title=R--}}
\end{{popupmenu}}

\begin{{insDLJS}}[AeBMenu]{{md}}{{Menu Data}}
\myMenu
\end{{insDLJS}}

    \makeatletter
        \let\@oddfoot\@empty\let\@evenfoot\@empty
        \def\@evenhead{{\thepage\hfil\slshape\leftmark
                        {{\rule[-0.11cm]{{-\textwidth}}{{0.03cm}}
                        \rule[-0.11cm]{{\textwidth}}{{0.03cm}}}}}}
        \def\@oddhead{{{{\slshape\rightmark}}\hfil\thepage
                        {{\rule[-0.11cm]{{-\textwidth}}{{0.03cm}}
                        \rule[-0.11cm]{{\textwidth}}{{0.03cm}}}}}}
        \let\@mkboth\markboth
        \markright{{{{\bf {replacement} }}\hskip 3em  \today}}
        \def\maketitle{{
            \centerline{{\Large\bfseries\@title}}
            \bigskip
        }}
    \makeatother

    \lstset{{language=python,
            extendedchars=true,
            aboveskip = 0.5ex,
            belowskip = 0.6ex,
            basicstyle=\ttfamily,
            keywordstyle=\sffamily\bfseries,
            identifierstyle=\sffamily,
            commentstyle=\slshape\color{{darkgreen}},
            stringstyle=\rmfamily\color{{blue}},
            showstringspaces=false,
            tabsize=4,
            breaklines=true,
            numberstyle=\footnotesize\color{{grey}},
            classoffset=1,
            morekeywords={{eyes,zeros,zeros_like,ones,ones_like,array,rand,indentity,mat,vander}},keywordstyle=\color{{darkblue}},
            classoffset=2,
            otherkeywords={{[,],=,:}},keywordstyle=\color{{purple}}\bfseries,
            classoffset=0""".format(andrewid=data['ANDREWID'],
            assignment=data['ASSIGNMENT'],
            fullname=data['NAME'],
            replacement=( re.sub( "_", r'\\_', options.infilename) )) + options.latexescapes * r""",
            mathescape=true""" +"""
            }
    """

        if options.nocode:
            latex_column_sep = r"""
    \setlength\columnseprule{0.4pt}
    """
        else:
            latex_column_sep = ""


        latex_doublepage = r"""
    \usepackage[landscape,left=1.5cm,right=1.1cm,top=1.8cm,bottom=1.2cm]{geometry}
    \usepackage{multicol}
    \def\inputBlocksize{\small}
    \makeatletter
        \renewcommand\normalsize{%
        \@setfontsize\normalsize\@ixpt\@xipt%
        \abovedisplayskip 8\p@ \@plus4\p@ \@minus4\p@
        \abovedisplayshortskip \z@ \@plus3\p@
        \belowdisplayshortskip 5\p@ \@plus3\p@ \@minus3\p@
        \belowdisplayskip \abovedisplayskip
        \let\@listi\@listI}
        \normalsize
        \renewcommand\small{%
        \@setfontsize\small\@viiipt\@ixpt%
        \abovedisplayskip 5\p@ \@plus2\p@ \@minus2\p@
        \abovedisplayshortskip \z@ \@plus1\p@
        \belowdisplayshortskip 3\p@ \@plus\p@ \@minus2\p@
        \def\@listi{\leftmargin\leftmargini
                    \topsep 3\p@ \@plus\p@ \@minus\p@
                    \parsep 2\p@ \@plus\p@ \@minus\p@
                    \itemsep \parsep}%
        \belowdisplayskip \abovedisplayskip
        }
        \renewcommand\footnotesize{%
        \@setfontsize\footnotesize\@viipt\@viiipt
        \abovedisplayskip 4\p@ \@plus2\p@ \@minus2\p@
        \abovedisplayshortskip \z@ \@plus1\p@
        \belowdisplayshortskip 2.5\p@ \@plus\p@ \@minus\p@
        \def\@listi{\leftmargin\leftmargini
                    \topsep 3\p@ \@plus\p@ \@minus\p@
                    \parsep 2\p@ \@plus\p@ \@minus\p@
                    \itemsep \parsep}%
        \belowdisplayskip \abovedisplayskip
        }
        \renewcommand\scriptsize{\@setfontsize\scriptsize\@vipt\@viipt}
        \renewcommand\tiny{\@setfontsize\tiny\@vpt\@vipt}
        \renewcommand\large{\@setfontsize\large\@xpt\@xiipt}
        \renewcommand\Large{\@setfontsize\Large\@xipt{13}}
        \renewcommand\LARGE{\@setfontsize\LARGE\@xiipt{14}}
        \renewcommand\huge{\@setfontsize\huge\@xivpt{18}}
        \renewcommand\Huge{\@setfontsize\Huge\@xviipt{22}}
        \setlength\parindent{14pt}
        \setlength\smallskipamount{3\p@ \@plus 1\p@ \@minus 1\p@}
        \setlength\medskipamount{6\p@ \@plus 2\p@ \@minus 2\p@}
        \setlength\bigskipamount{12\p@ \@plus 4\p@ \@minus 4\p@}
        \setlength\headheight{12\p@}
        \setlength\headsep   {25\p@}
        \setlength\topskip   {9\p@}
        \setlength\footskip{30\p@}
        \setlength\maxdepth{.5\topskip}
    \makeatother

    \AtBeginDocument{
    \setlength\columnsep{1.1cm}
    """ + latex_column_sep + r"""
    \begin{multicols*}{2}
    \small}
    \AtEndDocument{\end{multicols*}}
    """

        if options.double:
            self.preamble += latex_doublepage
        else:
            pass
            self.preamble += r"""\usepackage[top=1in,bottom=1in,left=1in,right=1in]{geometry}
    \def\inputBlocksize{\normalsize}

        """    

        if options.outtype == "tex":
            self.compile = self.compile2tex
        else:
            self.compile = self.compile2pdf


    def compile2tex(self, output_list, fileobject, options):
        global user_data_string
        """ Compiles the output_list to the tex file given the filename
        """
        tex_string = rst2latex(self.blocks2rst_string(output_list))

        # here we make it letter paper and change how the pdf opens in full width
        tex_string = tex_string.replace('\documentclass[a4paper]{article}',
                                        '''
\documentclass[pdfstartview=FitH,
               pdfproducer=pycse-publish-v{VERSION},
               pdfauthor={author},
               pdftitle={assignment}]{{article}}'''.format(VERSION=VERSION,
                                                           author=data['NAME'],
                                                           assignment=data['ASSIGNMENT']))
        tex_string = re.sub(r"\\begin{document}", 
                        protect(self.preamble) + r"\\begin{document}", tex_string)
        tex_string = re.sub(self.empty_listing, "", tex_string)

        
        # if there is user_data we insert it here.
        # the option --no-user suppresses this insertion
        tex_string = re.sub(r'\\end{document}', "", tex_string)
        path, fname = os.path.split(options.infilename)
        if user_data_string:
            tex_string += user_data_string
            tex_string +=  r'''
\vspace{{2mm}}            
\attachfile[description={0}]{{{0}}}
\vspace{{8mm}}

\PushButton[name=mymenu,
onclick={{var cChoice = \popUpMenu(myMenu);
this.info.Grade = cChoice;
var grade = this.getField("myGrade");
grade.value = cChoice;
        }}]{{Press to select grade}} Grade: \textField
	[\BC{{0 0 1}}\BG{{0.98 0.92 0.73}}
	\textColor{{1 0 0}}
	]{{myGrade}}{{0.5in}}{{12bp}}

\end{{document}}'''.format(fname)
        else:
            tex_string += r'''
\vspace{{2mm}}
\attachfile[description={0}]{{{0}}}
\vspace{{2mm}}
\end{{document}}'''.format(fname)

        # this is an awful hack around docutils
        tex_string = re.sub('%%% User specified packages and stylesheets',
                            '''
\usepackage[pdftex]{web}
\screensize{11in}{8.5in}
\margins{0.25in}{0.25in}{0.25in}{0.25in}
\usepackage{eforms}
\usepackage{popupmenu}

\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue,pdfstartview=FitH}

\usepackage{bookmark}''', tex_string)
         
        print >>fileobject, tex_string

    def compile2pdf(self, output_list, fileobject, options):
        """ Compiles the output_list to the tex file given the filename
        """
        self.compile2tex( output_list, fileobject, options)
        fileobject.close()
        tex2pdf(options.outfilename, options)
        map(safe_unlink, self.figure_list)
        self.figure_list = ()

main.TexCompiler = MyTexCompiler

class PylabShow(object):
    """ Factory for creating a function to replace pylab.show .
"""
    figure_list = ()
    
    figure_extension = "eps"

    def _set_options(self,options):
        if not options.outfilename in set(("-", None)):
            self.basename = "%s_pyreport_" % os.path.splitext(
                        os.path.basename(options.infilename))[0]
        else:
            self.basename = "_pyreport_"
        # XXX: Use pylab's pdf output
        #if options.figuretype == "pdf":
        # self.figure_extension = "eps"
        #else:
        self.figure_extension = options.figuretype
        
    def __call__(self):
        figure_name = '%s%d.%s' % ( self.basename,
                                    len(self.figure_list),
                                    self.figure_extension )
        
        import pylab
        pylab.savefig(figure_name)

main.myshow = PylabShow()

        
# patch to capture pyplot.show
import matplotlib.pyplot
matplotlib.pyplot.show = main.myshow

original_savefig = matplotlib.pyplot.savefig

# patch to capture savefig
def mysave(*args, **kwargs):
    'wrap savefig for publish'

    # we use hashes of the files to determine this
    from hashlib import sha1
    def git_hash(figure_name):
        with open(figure_name) as f:
            data = f.read()
    
        s = sha1()
        s.update("blob %u\0" % len(data))
        s.update(data)
        return s.hexdigest()
    
    # catching a user call
    self = main.myshow
    
    figure_name = '%s%d.%s' % ( self.basename,
                                len(self.figure_list),
                                self.figure_extension )

    if '_pyreport_' in args[0]:
        #this is coming from show. 
        original_savefig(args[0])

        current_hashes = [git_hash(x) for x in self.figure_list]
        this_hash = git_hash(args[0])

        if this_hash in current_hashes:
            os.unlink(args[0])
        else:
            self.figure_list += (figure_name, )
            print "Here goes figure %s" % figure_name
        return
    
    # first save what the user wants
    original_savefig(*args, **kwargs)
    # now what we need for the output
    if 'fname' in kwargs:
        del kwargs['fname']
    else:
        args = args[1:]
    # try to save with all the user-defined args
    original_savefig(figure_name, *args, **kwargs)

    # now let us check if the figure is already saved
    current_hashes = [git_hash(x) for x in self.figure_list]

    this_hash = git_hash(figure_name)
    if not this_hash in current_hashes:
        self.figure_list += (figure_name, )
        print "Here goes figure %s" % figure_name
    else:
        os.unlink(figure_name)
    
matplotlib.pyplot.savefig = mysave

    
##################################################################
##################################################################
import argparse
from uuid import getnode as get_mac
import re, socket
import datetime
from pyreport import pyreport, options

data = {}
userid = os.environ.get('USER', 'no user found')
date_submitted = datetime.datetime.now()

# mac address of submitting computer
mac = get_mac()
try:
    hostname, aliases, ipaddr =  socket.gethostbyaddr(socket.gethostbyname(socket.gethostname()))
except:
    hostname, aliases, ipaddr = None, None, None

if ipaddr:
    ipaddr = ipaddr[0] # it is usually a list. I think it is ok to take the first element.

data = {'mac':mac,
        'hostname':hostname,
        'ipaddr':ipaddr,
        'userid':userid,
        'date_submitted':date_submitted}

user_data_string = r'''
\begin{{verbatim}}
hostname: {data[hostname]}
ipaddr:   {data[ipaddr]}
mac:      {data[mac]}
\end{{verbatim}}

'''.format(data=data)


PROPERTIES = ['COURSE',
              'ASSIGNMENT',
              'ANDREWID',              
              'NAME']

parser = argparse.ArgumentParser(description='submit your python script and output in tex, or pdf')

parser.add_argument('files', nargs='*',                    
                    help='scripts to submit')

parser.add_argument('-v', action='store_true', help='be verbose')
parser.add_argument('--tex', action='store_true', help='make tex file')
parser.add_argument('--no-user', action='store_true', help='do not check for compliance or put user data in pdf')
parser.add_argument('--version', action='store_true', help='print version')

def publish(args):
    '''args is one of two things:
    1. a string from an ipython magic method
    2. the output from argparse
    '''
    global data,user_data_string
    
    if isinstance(args, unicode):
        # magic method provides a unicode string.
        # we have to parse the string
        args = parser.parse_args(args.split())

    INPUT = args.files[0]
        
    if args.no_user:
        user_data_string = False
        for prop in PROPERTIES:
            data[prop] = None
        
        name, ext = os.path.splitext(INPUT)
        BASENAME = name
    else:
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
        
    myoptions = ['-l', #allow LaTeX literal comment
                 '-e', #allow LaTeX math mode escape in code wih dollar signs
                 ]
    
    if args.v:
        myoptions += ['-v']

    if args.tex:
        myoptions += ['-t', 'tex']
        myoptions += ['-o','{0}'.format(BASENAME),]
    else:
        myoptions += ['-o','{0}'.format(BASENAME),]

    opts, args = options.parse_options(myoptions)

    opts.update({'infilename':INPUT})

    default_options, _not_used = options.option_parser.parse_args(args =[])
    default_options.figure_type = 'png'

    pyreport.main(open(INPUT), overrides=opts)

    # clean up
    for fname in ['md.djs']:
        if os.path.exists(fname):
            os.unlink(fname)

    try:
        os.startfile(BASENAME + '.pdf')
    except:
        pass
        
##################################################################

if __name__ == '__main__':

    args = parser.parse_args()

    #if len(args.files) > 1:
    #    print 'You can only publish one file at a time! Please try again.'
    #    import sys; sys.exit()
    
    #publish(args)
    
    # this seems clunky, but it lets you run the script on many files 
    # because it uses a clean namespace each time. Otherwise, you get namespace
    # pollution that eventually causes an error.
    
    from subprocess import Popen, PIPE

    if args.version:
        print 'Version = {0}'.format(VERSION)
        import sys; sys.exit()

    
    for f in args.files:
        
        cmd = u'{0}'.format(f)
        if args.v:
            cmd += ' -v'
        if args.tex:
            cmd += ' --tex'
        if args.no_user:
            cmd += ' --no-user'
        code = '''from pycse.publish import publish
publish(u'{0}')'''.format(cmd)
    
    
        p = Popen('python', stdout=PIPE, stderr=PIPE, stdin=PIPE)
        out, err = p.communicate(code)
        print out, err
        



    
    




