import IPython
import tabulate


class Heading(object):
    def __init__(self, title, level=1, tags=(), properties=None):
        self.title = title
        self.level = level
        self.tags = tags
        self.properties = properties

    def _repr_org(self):
        s = '*' * self.level + ' ' + self.title
        if self.tags:
            s += f"  :{':'.join(self.tags)}:"
        if self.properties:
            s += '\n:PROPERTIES:\n'
            for key in self.properties:
                s += f':{key}: {self.properties[key]}\n'
            s += ':END:'
        return s + '\n'

    def _repr_html(self):
        '''HTML representation of a Heading'''
        return f"<h{self.level}>{self.title}</h{self.level}>"

    def _repr_mimebundle_(self, include, exclude, **kwargs):
        """
        repr_mimebundle should accept include, exclude and **kwargs
        """

        data = {'text/html': self._repr_html(),
                'text/org': self._repr_org()
                }
        if include:
            data = {k: v for (k, v) in data.items() if k in include}
        if exclude:
            data = {k: v for (k, v) in data.items() if k not in exclude}
        return data


class Keyword:
    '''Keyword(key, value) -> #+key: value'''
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def _repr_org(self):
        return f'#+{self.key}: {self.value}' + '\n'

    def _repr_mimebundle_(self, include, exclude, **kwargs):
        """
        repr_mimebundle should accept include, exclude and **kwargs
        """

        data = {'text/org': self._repr_org()}
        if include:
            data = {k: v for (k, v) in data.items() if k in include}
        if exclude:
            data = {k: v for (k, v) in data.items() if k not in exclude}
        return data


class Comment:
    '''Comment(text) -> # text'''
    def __init__(self, text):
        self.text = text

    def _repr_org(self):
        return f'# {self.text}' + '\n'

    def _repr_mimebundle_(self, include, exclude, **kwargs):
        """
        repr_mimebundle should accept include, exclude and **kwargs
        """

        data = {'text/org': self._repr_org()}
        if include:
            data = {k: v for (k, v) in data.items() if k in include}
        if exclude:
            data = {k: v for (k, v) in data.items() if k not in exclude}
        return data


class Org:
    '''Org(text) -> text'''
    def __init__(self, text):
        self.text = text

    def _repr_org(self):
        return self.text + '\n'

    def _repr_mimebundle_(self, include, exclude, **kwargs):
        """
        repr_mimebundle should accept include, exclude and **kwargs
        """

        data = {'text/org': self._repr_org()}
        if include:
            data = {k: v for (k, v) in data.items() if k in include}
        if exclude:
            data = {k: v for (k, v) in data.items() if k not in exclude}
        return data    


class Figure:
    def __init__(self, fname, caption=None, name=None, attributes=()):
        self.fname = fname
        self.caption = caption
        self.name = name
        self.attributes = attributes

    def _repr_org(self):
        s = []
        for backend, attrs in self.attributes:
            s += [f'#+attr_{backend}: {attrs}']

        if self.name:
            s += [f'#+name: {self.name}']

        if self.caption:
            s += [f'#+caption: {self.caption}']

        s += [f'[[{self.fname}]]']

        return '\n'.join(s) + '\n'

    def _repr_mimebundle_(self, include, exclude, **kwargs):
        """
        repr_mimebundle should accept include, exclude and **kwargs
        """

        data = {'text/org': self._repr_org()}
        if include:
            data = {k: v for (k, v) in data.items() if k in include}
        if exclude:
            data = {k: v for (k, v) in data.items() if k not in exclude}
        return data


class Table:
    def __init__(self, data, headers=None, caption=None,
                 name=None, attributes=()):
        self.data = data
        self.headers = headers
        self.caption = caption
        self.name = name
        self.attributes = attributes

    def _repr_org(self):
        s = []
        for backend, attributes in self.attributes:
            s += [f'#+attr_{backend}: {attributes}']

        if self.name:
            s += [f'#+name: {self.name}']

        if self.caption:
            s += [f'#+caption: {self.caption}']

        s += [tabulate.tabulate(self.data, self.headers, tablefmt='orgtbl')]

        return '\n'.join(s)

    def _repr_mimebundle_(self, include, exclude, **kwargs):
        """
        repr_mimebundle should accept include, exclude and **kwargs
        """

        data = {'text/org': self._repr_org()}
        if include:
            data = {k: v for (k, v) in data.items() if k in include}
        if exclude:
            data = {k: v for (k, v) in data.items() if k not in exclude}
        return data


# * Rich displays for org-mode

class OrgFormatter(IPython.core.formatters.BaseFormatter):
    format_type = IPython.core.formatters.Unicode('text/org')
    print_method = IPython.core.formatters.ObjectName('_repr_org_')

    
ip = get_ipython()
ip.display_formatter.formatters['text/org'] = OrgFormatter()
ytv_f = ip.display_formatter.formatters['text/org']
ytv_f.for_type_by_name('IPython.lib.display', 'YouTubeVideo',
                       lambda V: f'{V.src}')    
