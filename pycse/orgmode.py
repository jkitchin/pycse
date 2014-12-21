'''Module to redirect stderr to stdout for use in Emacs + orgmode.

org-mode does not capture stderr in code blocks. Here we redirect it so
that the stderr stream shows up in the results section of a code
block.
'''

import sys
sys.stderr = sys.stdout
