'''Module to redirect stderr to stdout for use in Emacs + orgmode'''

import sys
sys.stderr = sys.stdout
