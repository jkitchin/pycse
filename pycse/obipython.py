"""Legacy module for ob-ipython and org-mode.

I should probably remove this when we switch to emacs-jupyter.
"""

import IPython
from tabulate import tabulate


class OrgFormatter(IPython.core.formatters.BaseFormatter):
    """A special formatter for Org."""

    def __call__(self, obj):
        """Call function for the class."""
        try:
            return tabulate(
                obj, headers="keys", tablefmt="orgtbl", showindex="always"
            )
        # I am not sure what exceptions get thrown, or why this is here.
        except:  # noqa: E722
            return None


try:
    ip = IPython.get_ipython()
    ip.display_formatter.formatters["text/org"] = OrgFormatter()
except AttributeError:
    pass
