"""Provides classes to convert python data into org markup."""
import IPython
import tabulate


class Heading:
    """An orgmode headline."""

    def __init__(self, title, level=1, tags=(), properties=None):
        """Initialize a Heading."""
        self.title = title
        self.level = level
        self.tags = tags
        self.properties = properties

    def _repr_org(self):
        """Provide an org representation."""
        s = "*" * self.level + " " + self.title
        if self.tags:
            s += f"  :{':'.join(self.tags)}:"
        if self.properties:
            s += "\n:PROPERTIES:\n"
            for key in self.properties:
                s += f":{key}: {self.properties[key]}\n"
            s += ":END:"
        return s + "\n"

    def _repr_html(self):
        """HTML representation of a Heading."""
        return f"<h{self.level}>{self.title}</h{self.level}>"

    def _repr_mimebundle_(self, include, exclude, **kwargs):
        """Mimebundle representation of a Heading.

        repr_mimebundle should accept include, exclude and **kwargs.
        """
        data = {"text/html": self._repr_html(), "text/org": self._repr_org()}
        if include:
            data = {k: v for (k, v) in data.items() if k in include}
        if exclude:
            data = {k: v for (k, v) in data.items() if k not in exclude}
        return data


class Keyword:
    """Keyword(key, value) -> #+key: value."""

    def __init__(self, key, value):
        """Initialize a Keyword."""
        self.key = key
        self.value = value

    def _repr_org(self):
        """Provide org representation."""
        return f"#+{self.key}: {self.value}" + "\n"

    def _repr_mimebundle_(self, include, exclude, **kwargs):
        """Provide a mimebundle representation.

        repr_mimebundle should accept include, exclude and **kwargs
        """
        data = {"text/org": self._repr_org()}
        if include:
            data = {k: v for (k, v) in data.items() if k in include}
        if exclude:
            data = {k: v for (k, v) in data.items() if k not in exclude}
        return data


class Comment:
    """Comment(text) -> # text."""

    def __init__(self, text):
        """Initialize a comment."""
        self.text = text

    def _repr_org(self):
        """Provide an org representation."""
        return f"# {self.text}" + "\n"

    def _repr_mimebundle_(self, include, exclude, **kwargs):
        """Provide a mimebundle representation.

        repr_mimebundle should accept include, exclude and **kwargs
        """
        data = {"text/org": self._repr_org()}
        if include:
            data = {k: v for (k, v) in data.items() if k in include}
        if exclude:
            data = {k: v for (k, v) in data.items() if k not in exclude}
        return data


class Org:
    """Org(text) -> text."""

    def __init__(self, text):
        """Initialize an Org object."""
        self.text = text

    def _repr_org(self):
        """Provide an org representation."""
        return self.text + "\n"

    def _repr_mimebundle_(self, include, exclude, **kwargs):
        """Provide a mimebundle representation.

        repr_mimebundle should accept include, exclude and **kwargs
        """
        data = {"text/org": self._repr_org()}
        if include:
            data = {k: v for (k, v) in data.items() if k in include}
        if exclude:
            data = {k: v for (k, v) in data.items() if k not in exclude}
        return data


class Figure:
    """A Figure class for org.

    It combines a filename, caption, name and attributes.
    """

    def __init__(self, fname, caption=None, name=None, attributes=()):
        """Initialize a figure."""
        self.fname = fname
        self.caption = caption
        self.name = name
        self.attributes = attributes

    def _repr_org(self):
        s = []
        for backend, attrs in self.attributes:
            s += [f"#+attr_{backend}: {attrs}"]

        if self.name:
            s += [f"#+name: {self.name}"]

        if self.caption:
            s += [f"#+caption: {self.caption}"]

        s += [f"[[{self.fname}]]"]

        return "\n".join(s) + "\n"

    def _repr_mimebundle_(self, include, exclude, **kwargs):
        """Provide a mimebundle representation.

        repr_mimebundle should accept include, exclude and **kwargs.
        """
        data = {"text/org": self._repr_org()}
        if include:
            data = {k: v for (k, v) in data.items() if k in include}
        if exclude:
            data = {k: v for (k, v) in data.items() if k not in exclude}
        return data


class Table:
    """A Table object for org."""

    def __init__(
        self, data, headers=None, caption=None, name=None, attributes=()
    ):
        """Initialize a table."""
        self.data = data
        self.headers = headers
        self.caption = caption
        self.name = name
        self.attributes = attributes

    def _repr_org(self):
        """Provide an org representation."""
        s = []
        for backend, attributes in self.attributes:
            s += [f"#+attr_{backend}: {attributes}"]

        if self.name:
            s += [f"#+name: {self.name}"]

        if self.caption:
            s += [f"#+caption: {self.caption}"]

        s += [tabulate.tabulate(self.data, self.headers, tablefmt="orgtbl")]

        return "\n".join(s)

    def _repr_mimebundle_(self, include, exclude, **kwargs):
        """Provide a mimebundle representation.

        repr_mimebundle should accept include, exclude and **kwargs
        """
        data = {"text/org": self._repr_org()}
        if include:
            data = {k: v for (k, v) in data.items() if k in include}
        if exclude:
            data = {k: v for (k, v) in data.items() if k not in exclude}
        return data


# * Rich displays for org-mode


class OrgFormatter(IPython.core.formatters.BaseFormatter):
    """A special formatter for org objects."""

    format_type = IPython.core.formatters.Unicode("text/org")
    print_method = IPython.core.formatters.ObjectName("_repr_org_")


try:
    ip = get_ipython()
    ip.display_formatter.formatters["text/org"] = OrgFormatter()
    ytv_f = ip.display_formatter.formatters["text/org"]
    ytv_f.for_type_by_name(
        "IPython.lib.display", "YouTubeVideo", lambda V: f"{V.src}"
    )
# get_ipython is not defined for tests I think.
except NameError:
    pass
