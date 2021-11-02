"""Module to enhance Python for use in org-mode.

1. redirect stderr to stdout for use in Emacs + orgmode.

   org-mode does not capture stderr in code blocks. Here we redirect it so that
   the stderr stream shows up in the results section of a code block.

2. Modify matplotlib.pyplot.savefig and show

3. Provide table and figure commands that generate org-markup.

4. Provide functions that generate org markup, e.g. results, comments,
headlines and links.

# Copyright 2015, John Kitchin
# (see accompanying license files for details).

"""


# Matplotlib modifications
import io
import os
import matplotlib.pyplot
from hashlib import sha1
import sys


def stderr_to_stdout():
    """Redirect stderr to stdout so it can be captured.

    Note: if your python returns an error, you may see nothing from this.

    """
    import sys
    sys.stderr = sys.stdout


class print_redirect:
    '''Context manager for pycse.orgmode
    Most functions print to stdout. Use this to capture it in a file.
    with print_redirect(some_file):
        org()
        print(something)

    mode='w' overwrites the file
    mode='a' appends to the file
    '''
    def __init__(self, fname, mode='w'):
        self.fname = fname
        self.mode = mode

    def __enter__(self):
        sys.stdout = open(self.fname, self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = sys.__stdout__


original_savefig = matplotlib.pyplot.savefig


# patch to capture savefig
def mysave(fname, *args, **kwargs):
    """wrap savefig for org-mode.

    Returns an org-mode link to the file.
    """

    original_savefig(fname, *args, **kwargs)
    return '[[file:{}]]'.format(fname)


matplotlib.pyplot.savefig = mysave


def git_hash(string):
    """Return a git hash of string."""

    s = sha1()
    g = "blob {0:d}\0".format(len(string))
    s.update(g.encode('utf-8'))
    s.update(string)
    return s.hexdigest()


original_show = matplotlib.pyplot.show

SHOW = True


def myshow(*args, **kwargs):
    """Wrap matplotlib.pyplot.show for orgmode

    Saves the figure in a directory called pyshow with the filename derived
    from its git-hash.

    """
    format = "png"
    sio = io.BytesIO()
    original_savefig(sio, format=format)
    fig_contents = sio.getvalue()

    hash = git_hash(fig_contents)

    if not os.path.isdir('pyshow'):
        os.mkdir('pyshow')

    png = os.path.join('pyshow', hash + '.png')

    with open(png, 'wb') as f:
        f.write(fig_contents)

    print('[[file:{}]]'.format(png))

    if SHOW:
        original_show(*args, **kwargs)


matplotlib.pyplot.show = myshow


# Tables and figures
def table(data, name=None,
          caption=None, attributes=None,
          none=''):
    """Return a formatted table.

    :data: A list-like data structure. A None value is converted to hline.
    :name: The name of the table
    :caption: The caption text
    :attributes: [(backend, 'attributes')]
    :none: A string for None values

    """
    s = []

    if caption is not None:
        s += ['#+caption: {}'.format(caption)]

    if attributes is not None:
        for backend, attrs in attributes:
            s += ['#+attr_{}: {}'.format(backend.lower(), attrs)]

    if name is not None:
        s += ['#+name: {}'.format(name)]

    for row in data:
        if row is None:
            s += ['|-']
        else:
            s += ['| ' + ' | '.join([str(x) if x is not None else none
                                     for x in row]) + '|']

    print('\n'.join(s))


def figure(fname, caption=None, name=None, attributes=None):
    """Return a formatted figure.

    :fname: A string for the filename.
    :caption: A string of the caption text.
    :name: A string for a label.
    :attributes: [(backend, 'attributes')]

    """
    s = []

    if attributes is not None:
        for backend, attrs in attributes:
            s += ['#+attr_{}: {}'.format(backend.lower(), attrs)]

    if name is not None:
        s += ['#+name: {}'.format(name)]

    if caption is not None:
        s += ['#+caption: {}'.format(caption)]

    if fname.startswith('[[file:'):
        s += [fname]
    else:
        if not os.path.exists(fname):
            if not os.path.exists(os.path.dirname(fname)):
                os.makedirs(os.path.dirname(fname), exist_ok=True)
        s += [mysave(fname)]

    print('\n'.join(s))


def verbatim(s):
    """Print s in verbatim.

    If s is one line, print it in ==, otherwise use a block.
    """
    if '\n' in str(s):
        print('\n#+begin_example\n{}\n#+end_example\n'.format(s))
    else:
        print('={}='.format(s))


def comment(s):
    """Print s as a comment

    If s is one line, print it in #, otherwise use a block.
    """
    if '\n' in str(s):
        print('\n#+begin_comment\n{}\n#+end_comment\n'.format(s))
    else:
        import textwrap
        print(textwrap.fill(s, initial_indent='# ',
                            subsequent_indent='# ',
                            width=79))


def fixed_width(s):
    """Print s as a fixed-width element."""
    print('\n'.join([': ' + x for x in str(s).split('\n')]))


def result(s):
    """Convenience for fixed_width. An org src block result."""
    return fixed_width(str(s))


def latex(s):
    """Print s as a latex block."""
    print('\n#+begin_latex\n{}\n#+end_latex\n'.format(s))


def org(s):
    """Print s as it is."""
    print(s)


def headline(headline, level=1,
             todo=None, tags=(),
             deadline=None,
             scheduled=None,
             properties=None,
             body=None):
    """Print an org headline.

    :headline: A string for the headline

    :level: an integer for number of stars in the headline

    :tags: a list of strings as tags for the headline
    :properties: A dictionary of property: value pairs
    :body: a string representing the body.

    """
    s = '*' * level + ' '
    if todo is not None:
        s += '{} '.format(todo)

    s += headline
    if tags:
        s += ' :' + ":".join(tags) + ':'
    s += '\n'

    if scheduled and deadline:
        s += '  SCHEDULED: {} DEADLINE: {}\n'.format(scheduled,
                                                     deadline)
    elif scheduled:
        s += '  SCHEDULED: {}\n'.format(scheduled)
    elif deadline:
        s += '  DEADLINE: {}\n'.format(deadline)
    if properties:
        s += '  :PROPERTIES:\n'
        for key, val in properties.items():
            s += '  :{}: {}\n'.format(key, val)
        s += '  :END:\n\n'

    if body:
        s += body + '\n'

    print(s)


def link(type=None,
         path=None,
         desc=None):
    """Print an org link [[type:path][desc]].

    :path: is all that is mandatory.
    """

    s = '[['
    if type is not None:
        s += type + ':'
    s += path + ']'

    if desc is not None:
        s += '[{}]'.format(desc)

    s += ']'

    print(s)
