from nose.tools import raises
from pycse.orgmode import *
from io import StringIO
import sys
sys.stdout = StringIO()

def test_link_1():
    link('file', 'path.org')
    s= sys.stdout.getvalue()
    print(s, s == '[[file:path.org]]\n')
    assert s == '[[file:path.org]]\n'

def test_link_2():
    link('file', 'path.org', 'desc')
    assert sys.stdout.getvalue() == '[[file:path.org][desc]]\n'

def test_link_3():
    link(path="http://test.org")
    assert sys.stdout.getvalue() == '[[http://test.org]]\n'


def test_headline_1():
    headline("test")
    assert sys.stdout.getvalue() == "* test\n\n"

def test_hl_2():
    headline("test", 2)
    assert sys.stdout.getvalue() == "** test\n\n"

def test_hl_3():
    headline("test", scheduled='test')
    assert sys.stdout.getvalue() == "* test\n  SCHEDULED: test\n\n"

def test_hl_4():
    headline("test", scheduled='test', deadline='test')
    assert sys.stdout.getvalue() == "* test\n  SCHEDULED: test DEADLINE: test\n\n"

def test_hl_5():
    headline("test", deadline='test')
    assert sys.stdout.getvalue() == "* test\n  DEADLINE: test\n\n"

def test_hl_tags():
    headline("test", tags=['a', 'b'])
    assert sys.stdout.getvalue() == "* test :a:b:\n\n"


def test_hl_6():
    headline("test", body="some text")
    assert sys.stdout.getvalue() == '* test\nsome text\n\n'

def test_hl_7():
    headline("test", properties={"one": "two"})
    assert sys.stdout.getvalue() == "* test\n  :PROPERTIES:\n  :one: two\n  :END:\n\n\n"

def test_org():
    org("test")
    assert sys.stdout.getvalue() == "test\n"

def test_latex():
    latex("test")
    assert sys.stdout.getvalue() == "\n#+BEGIN_LATEX\ntest\n#+END_LATEX\n\n"

def test_fw():
    fixed_width("test\ntest")
    assert sys.stdout.getvalue() == ": test\n: test\n"

def test_comment():
    comment("test")
    assert sys.stdout.getvalue() == "# test\n"

def test_comment_2():
    comment("test\ntest")
    assert sys.stdout.getvalue() == "\n#+BEGIN_COMMENT\ntest\ntest\n#+END_COMMENT\n\n"

def test_verb():
    verbatim("test")
    assert sys.stdout.getvalue() == "=test=\n"

def test_verb_2():
    verbatim("test\ntest")
    assert sys.stdout.getvalue() == "\n#+BEGIN_EXAMPLE\ntest\ntest\n#+END_EXAMPLE\n\n"

@raises(Exception)
def test_fig():
    figure("test")
    assert sys.stdout.getvalue() == '[[file:test]]\n'

def test_fig2():
    figure("pycse/tests/test_orgmode.py")
    assert sys.stdout.getvalue() == '[[file:pycse/tests/test_orgmode.py]]\n'

def test_fig3():
    figure("pycse/tests/test_orgmode.py", caption="test")
    assert sys.stdout.getvalue() == '#+CAPTION: test\n[[file:pycse/tests/test_orgmode.py]]\n'

def test_fig4():
    figure("pycse/tests/test_orgmode.py", caption="test", name='fig')
    assert sys.stdout.getvalue() == '#+NAME: fig\n#+CAPTION: test\n[[file:pycse/tests/test_orgmode.py]]\n'

def test_fig5():
    figure("pycse/tests/test_orgmode.py", caption="test", name='fig', attributes=[('org', ':width 300')])
    assert sys.stdout.getvalue() == '#+ATTR_org: :width 300\n#+NAME: fig\n#+CAPTION: test\n[[file:pycse/tests/test_orgmode.py]]\n'

def test_tab_1():
    table([[0, 1]])
    assert sys.stdout.getvalue() == '| 0 | 1|\n'

def test_tab_2():
    table([[0, 1]], caption='test')
    assert sys.stdout.getvalue() == '#+CAPTION: test\n| 0 | 1|\n'
