"""Provides classes to convert python data into org markup.

This module provides classes for generating Emacs Org-mode markup from Python
data structures. Each class implements _repr_org() for org-mode representation
and _repr_mimebundle_() for Jupyter notebook display.
"""

import IPython
import tabulate


def _filter_mimebundle(data, include, exclude):
    """Filter mimebundle data based on include/exclude lists.

    Parameters
    ----------
    data : dict
        The mimebundle data dictionary
    include : set or None
        If provided, only include these MIME types
    exclude : set or None
        If provided, exclude these MIME types

    Returns
    -------
    dict
        Filtered mimebundle data
    """
    if include is not None:
        data = {k: v for k, v in data.items() if k in include}
    if exclude is not None:
        data = {k: v for k, v in data.items() if k not in exclude}
    return data


class Heading:
    """An Org-mode headline with optional tags and properties.

    Parameters
    ----------
    title : str
        The headline text
    level : int, optional
        Heading level (number of stars), default=1
    tags : tuple of str, optional
        Tags for the headline
    properties : dict, optional
        Property drawer key-value pairs
    """

    def __init__(self, title, level=1, tags=(), properties=None):
        """Initialize a Heading."""
        self.title = title
        self.level = max(1, int(level))  # Ensure level is at least 1
        self.tags = tuple(tags) if tags else ()
        self.properties = properties or {}

    def _repr_org(self):
        """Provide an org representation."""
        s = "*" * self.level + " " + self.title
        if self.tags:
            s += f"  :{':'.join(self.tags)}:"
        if self.properties:
            s += "\n:PROPERTIES:\n"
            for key, value in self.properties.items():
                s += f":{key}: {value}\n"
            s += ":END:"
        return s + "\n"

    def __str__(self):
        """Return org representation as string."""
        return self._repr_org()

    def _repr_html(self):
        """HTML representation of a Heading."""
        return f"<h{self.level}>{self.title}</h{self.level}>"

    def _repr_mimebundle_(self, include=None, exclude=None, **kwargs):
        """Mimebundle representation of a Heading.

        Parameters
        ----------
        include : set, optional
            MIME types to include
        exclude : set, optional
            MIME types to exclude
        **kwargs
            Additional keyword arguments (ignored)

        Returns
        -------
        dict
            Mimebundle data dictionary
        """
        data = {"text/html": self._repr_html(), "text/org": self._repr_org()}
        return _filter_mimebundle(data, include, exclude)


class Keyword:
    """An Org-mode keyword (#+KEY: value).

    Parameters
    ----------
    key : str
        The keyword name (e.g., 'TITLE', 'AUTHOR')
    value : str
        The keyword value
    """

    def __init__(self, key, value):
        """Initialize a Keyword."""
        self.key = key
        self.value = value

    def _repr_org(self):
        """Provide org representation."""
        return f"#+{self.key}: {self.value}\n"

    def __str__(self):
        """Return org representation as string."""
        return self._repr_org()

    def _repr_mimebundle_(self, include=None, exclude=None, **kwargs):
        """Provide a mimebundle representation.

        Parameters
        ----------
        include : set, optional
            MIME types to include
        exclude : set, optional
            MIME types to exclude
        **kwargs
            Additional keyword arguments (ignored)

        Returns
        -------
        dict
            Mimebundle data dictionary
        """
        data = {"text/org": self._repr_org()}
        return _filter_mimebundle(data, include, exclude)


class Comment:
    """An Org-mode comment line (# text).

    Parameters
    ----------
    text : str
        The comment text
    """

    def __init__(self, text):
        """Initialize a comment."""
        self.text = text

    def _repr_org(self):
        """Provide an org representation."""
        return f"# {self.text}\n"

    def __str__(self):
        """Return org representation as string."""
        return self._repr_org()

    def _repr_mimebundle_(self, include=None, exclude=None, **kwargs):
        """Provide a mimebundle representation.

        Parameters
        ----------
        include : set, optional
            MIME types to include
        exclude : set, optional
            MIME types to exclude
        **kwargs
            Additional keyword arguments (ignored)

        Returns
        -------
        dict
            Mimebundle data dictionary
        """
        data = {"text/org": self._repr_org()}
        return _filter_mimebundle(data, include, exclude)


class Org:
    """Raw Org-mode text passthrough.

    This class wraps arbitrary Org-mode markup text.

    Parameters
    ----------
    text : str
        The org-mode text
    """

    def __init__(self, text):
        """Initialize an Org object."""
        self.text = text

    def _repr_org(self):
        """Provide an org representation."""
        return self.text if self.text.endswith("\n") else self.text + "\n"

    def __str__(self):
        """Return org representation as string."""
        return self._repr_org()

    def _repr_mimebundle_(self, include=None, exclude=None, **kwargs):
        """Provide a mimebundle representation.

        Parameters
        ----------
        include : set, optional
            MIME types to include
        exclude : set, optional
            MIME types to exclude
        **kwargs
            Additional keyword arguments (ignored)

        Returns
        -------
        dict
            Mimebundle data dictionary
        """
        data = {"text/org": self._repr_org()}
        return _filter_mimebundle(data, include, exclude)


class Figure:
    """An Org-mode figure with optional caption, name, and attributes.

    Parameters
    ----------
    fname : str
        The filename or path to the figure
    caption : str, optional
        Figure caption
    name : str, optional
        Figure name (for cross-referencing)
    attributes : tuple of (backend, attrs), optional
        Backend-specific attributes (e.g., ('html', ':width 500'))
    """

    def __init__(self, fname, caption=None, name=None, attributes=()):
        """Initialize a figure."""
        self.fname = fname
        self.caption = caption
        self.name = name
        self.attributes = tuple(attributes) if attributes else ()

    def _repr_org(self):
        """Provide org representation."""
        lines = []
        for backend, attrs in self.attributes:
            lines.append(f"#+attr_{backend}: {attrs}")

        if self.name:
            lines.append(f"#+name: {self.name}")

        if self.caption:
            lines.append(f"#+caption: {self.caption}")

        lines.append(f"[[{self.fname}]]")

        return "\n".join(lines) + "\n"

    def __str__(self):
        """Return org representation as string."""
        return self._repr_org()

    def _repr_mimebundle_(self, include=None, exclude=None, **kwargs):
        """Provide a mimebundle representation.

        Parameters
        ----------
        include : set, optional
            MIME types to include
        exclude : set, optional
            MIME types to exclude
        **kwargs
            Additional keyword arguments (ignored)

        Returns
        -------
        dict
            Mimebundle data dictionary
        """
        data = {"text/org": self._repr_org()}
        return _filter_mimebundle(data, include, exclude)


class Table:
    """An Org-mode table with optional headers, caption, name, and attributes.

    Parameters
    ----------
    data : list of lists
        The table data (rows)
    headers : list, optional
        Column headers
    caption : str, optional
        Table caption
    name : str, optional
        Table name (for cross-referencing)
    attributes : tuple of (backend, attrs), optional
        Backend-specific attributes
    """

    def __init__(self, data, headers=None, caption=None, name=None, attributes=()):
        """Initialize a table."""
        self.data = data
        self.headers = headers
        self.caption = caption
        self.name = name
        self.attributes = tuple(attributes) if attributes else ()

    def _repr_org(self):
        """Provide an org representation."""
        lines = []
        for backend, attrs in self.attributes:
            lines.append(f"#+attr_{backend}: {attrs}")

        if self.name:
            lines.append(f"#+name: {self.name}")

        if self.caption:
            lines.append(f"#+caption: {self.caption}")

        # Handle None or empty data and headers
        data = self.data if self.data is not None else []
        headers = self.headers if self.headers is not None else []
        lines.append(tabulate.tabulate(data, headers, tablefmt="orgtbl"))

        return "\n".join(lines) + "\n"

    def __str__(self):
        """Return org representation as string."""
        return self._repr_org()

    def _repr_mimebundle_(self, include=None, exclude=None, **kwargs):
        """Provide a mimebundle representation.

        Parameters
        ----------
        include : set, optional
            MIME types to include
        exclude : set, optional
            MIME types to exclude
        **kwargs
            Additional keyword arguments (ignored)

        Returns
        -------
        dict
            Mimebundle data dictionary
        """
        data = {"text/org": self._repr_org()}
        return _filter_mimebundle(data, include, exclude)


# * Rich displays for org-mode


class OrgFormatter(IPython.core.formatters.BaseFormatter):
    """Formatter for displaying org-mode objects in Jupyter notebooks.

    This formatter enables rich display of Org-mode content in Jupyter
    notebooks by registering the text/org MIME type.
    """

    format_type = IPython.core.formatters.Unicode("text/org")
    print_method = IPython.core.formatters.ObjectName("_repr_org_")


def _register_org_formatter():
    """Register the Org formatter with IPython if available.

    This function is called on module import to set up org-mode display
    support in Jupyter notebooks. It's safe to call in non-IPython contexts.
    """
    try:
        ip = get_ipython()
        ip.display_formatter.formatters["text/org"] = OrgFormatter()
        org_formatter = ip.display_formatter.formatters["text/org"]
        # Register YouTube video display in org format
        org_formatter.for_type_by_name("IPython.lib.display", "YouTubeVideo", lambda V: f"{V.src}")
    except NameError:
        # get_ipython is not defined outside of IPython/Jupyter
        pass


# Register the formatter on module import
_register_org_formatter()
