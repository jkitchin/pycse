"""Tests for orgmode module."""

from unittest.mock import MagicMock, patch

from pycse.orgmode import (
    Heading,
    Keyword,
    Comment,
    Org,
    Figure,
    Table,
    OrgFormatter,
    _filter_mimebundle,
)


class TestFilterMimebundle:
    """Tests for the _filter_mimebundle helper function."""

    def test_no_filters(self):
        """Test with no include or exclude filters."""
        data = {"text/html": "<p>test</p>", "text/org": "test", "text/plain": "test"}
        result = _filter_mimebundle(data, None, None)
        assert result == data

    def test_include_filter(self):
        """Test with include filter."""
        data = {"text/html": "<p>test</p>", "text/org": "test", "text/plain": "test"}
        result = _filter_mimebundle(data, {"text/html", "text/org"}, None)
        assert result == {"text/html": "<p>test</p>", "text/org": "test"}
        assert "text/plain" not in result

    def test_exclude_filter(self):
        """Test with exclude filter."""
        data = {"text/html": "<p>test</p>", "text/org": "test", "text/plain": "test"}
        result = _filter_mimebundle(data, None, {"text/plain"})
        assert result == {"text/html": "<p>test</p>", "text/org": "test"}
        assert "text/plain" not in result

    def test_both_filters(self):
        """Test with both include and exclude filters."""
        data = {"text/html": "<p>test</p>", "text/org": "test", "text/plain": "test"}
        # Include takes precedence, then exclude filters from that
        result = _filter_mimebundle(data, {"text/html", "text/org"}, {"text/html"})
        assert result == {"text/org": "test"}

    def test_empty_include(self):
        """Test with empty include set."""
        data = {"text/html": "<p>test</p>", "text/org": "test"}
        result = _filter_mimebundle(data, set(), None)
        assert result == {}

    def test_empty_exclude(self):
        """Test with empty exclude set."""
        data = {"text/html": "<p>test</p>", "text/org": "test"}
        result = _filter_mimebundle(data, None, set())
        assert result == data


class TestHeading:
    """Tests for the Heading class."""

    def test_basic_initialization(self):
        """Test basic heading initialization."""
        h = Heading("Test Heading")
        assert h.title == "Test Heading"
        assert h.level == 1
        assert h.tags == ()
        assert h.properties == {}

    def test_initialization_with_all_params(self):
        """Test heading with all parameters."""
        h = Heading(
            "Test Heading",
            level=2,
            tags=("tag1", "tag2"),
            properties={"CUSTOM_ID": "test-id", "CREATED": "2024-01-01"},
        )
        assert h.title == "Test Heading"
        assert h.level == 2
        assert h.tags == ("tag1", "tag2")
        assert h.properties == {"CUSTOM_ID": "test-id", "CREATED": "2024-01-01"}

    def test_level_minimum_enforced(self):
        """Test that level is at least 1."""
        h = Heading("Test", level=0)
        assert h.level == 1

        h = Heading("Test", level=-5)
        assert h.level == 1

    def test_level_float_conversion(self):
        """Test that level is converted to int."""
        h = Heading("Test", level=2.7)
        assert h.level == 2
        assert isinstance(h.level, int)

    def test_tags_conversion_to_tuple(self):
        """Test that tags are converted to tuple."""
        h = Heading("Test", tags=["tag1", "tag2"])
        assert h.tags == ("tag1", "tag2")
        assert isinstance(h.tags, tuple)

    def test_empty_tags(self):
        """Test with empty tags."""
        h = Heading("Test", tags=[])
        assert h.tags == ()

    def test_none_properties(self):
        """Test with None properties."""
        h = Heading("Test", properties=None)
        assert h.properties == {}

    def test_repr_org_simple(self):
        """Test _repr_org for simple heading."""
        h = Heading("Test Heading", level=1)
        assert h._repr_org() == "* Test Heading\n"

    def test_repr_org_multilevel(self):
        """Test _repr_org for different heading levels."""
        h = Heading("Level 3", level=3)
        assert h._repr_org() == "*** Level 3\n"

    def test_repr_org_with_tags(self):
        """Test _repr_org with tags."""
        h = Heading("Test", level=1, tags=("tag1", "tag2"))
        assert h._repr_org() == "* Test  :tag1:tag2:\n"

    def test_repr_org_with_properties(self):
        """Test _repr_org with properties."""
        h = Heading("Test", level=1, properties={"CUSTOM_ID": "test-id"})
        expected = "* Test\n:PROPERTIES:\n:CUSTOM_ID: test-id\n:END:\n"
        assert h._repr_org() == expected

    def test_repr_org_with_tags_and_properties(self):
        """Test _repr_org with both tags and properties."""
        h = Heading(
            "Test",
            level=2,
            tags=("python", "test"),
            properties={"CUSTOM_ID": "test-id", "CREATED": "2024-01-01"},
        )
        result = h._repr_org()
        assert result.startswith("** Test  :python:test:\n")
        assert ":PROPERTIES:" in result
        assert ":CUSTOM_ID: test-id" in result
        assert ":CREATED: 2024-01-01" in result
        assert ":END:" in result

    def test_str(self):
        """Test __str__ method."""
        h = Heading("Test", level=2, tags=("tag",))
        assert str(h) == h._repr_org()

    def test_repr_html(self):
        """Test _repr_html method."""
        h = Heading("Test Heading", level=1)
        assert h._repr_html() == "<h1>Test Heading</h1>"

        h = Heading("Test Heading", level=3)
        assert h._repr_html() == "<h3>Test Heading</h3>"

    def test_repr_mimebundle(self):
        """Test _repr_mimebundle method."""
        h = Heading("Test", level=2)
        bundle = h._repr_mimebundle_()
        assert "text/html" in bundle
        assert "text/org" in bundle
        assert bundle["text/html"] == "<h2>Test</h2>"
        assert bundle["text/org"] == "** Test\n"

    def test_repr_mimebundle_with_include(self):
        """Test _repr_mimebundle with include parameter."""
        h = Heading("Test")
        bundle = h._repr_mimebundle_(include={"text/org"})
        assert "text/org" in bundle
        assert "text/html" not in bundle

    def test_repr_mimebundle_with_exclude(self):
        """Test _repr_mimebundle with exclude parameter."""
        h = Heading("Test")
        bundle = h._repr_mimebundle_(exclude={"text/html"})
        assert "text/org" in bundle
        assert "text/html" not in bundle


class TestKeyword:
    """Tests for the Keyword class."""

    def test_initialization(self):
        """Test basic keyword initialization."""
        k = Keyword("TITLE", "My Document")
        assert k.key == "TITLE"
        assert k.value == "My Document"

    def test_repr_org(self):
        """Test _repr_org method."""
        k = Keyword("TITLE", "My Document")
        assert k._repr_org() == "#+TITLE: My Document\n"

        k = Keyword("AUTHOR", "John Doe")
        assert k._repr_org() == "#+AUTHOR: John Doe\n"

    def test_str(self):
        """Test __str__ method."""
        k = Keyword("OPTIONS", "toc:nil")
        assert str(k) == k._repr_org()

    def test_repr_mimebundle(self):
        """Test _repr_mimebundle method."""
        k = Keyword("TITLE", "Test")
        bundle = k._repr_mimebundle_()
        assert "text/org" in bundle
        assert bundle["text/org"] == "#+TITLE: Test\n"

    def test_repr_mimebundle_with_filters(self):
        """Test _repr_mimebundle with include/exclude."""
        k = Keyword("TITLE", "Test")
        bundle = k._repr_mimebundle_(include={"text/org"})
        assert "text/org" in bundle

        bundle = k._repr_mimebundle_(exclude={"text/org"})
        assert "text/org" not in bundle


class TestComment:
    """Tests for the Comment class."""

    def test_initialization(self):
        """Test comment initialization."""
        c = Comment("This is a comment")
        assert c.text == "This is a comment"

    def test_repr_org(self):
        """Test _repr_org method."""
        c = Comment("This is a comment")
        assert c._repr_org() == "# This is a comment\n"

    def test_str(self):
        """Test __str__ method."""
        c = Comment("Test comment")
        assert str(c) == c._repr_org()

    def test_repr_mimebundle(self):
        """Test _repr_mimebundle method."""
        c = Comment("Test")
        bundle = c._repr_mimebundle_()
        assert "text/org" in bundle
        assert bundle["text/org"] == "# Test\n"

    def test_empty_comment(self):
        """Test empty comment."""
        c = Comment("")
        assert c._repr_org() == "# \n"


class TestOrg:
    """Tests for the Org class."""

    def test_initialization(self):
        """Test Org initialization."""
        o = Org("Some org text")
        assert o.text == "Some org text"

    def test_repr_org_with_newline(self):
        """Test _repr_org when text has newline."""
        o = Org("Text with newline\n")
        assert o._repr_org() == "Text with newline\n"

    def test_repr_org_without_newline(self):
        """Test _repr_org when text lacks newline."""
        o = Org("Text without newline")
        assert o._repr_org() == "Text without newline\n"

    def test_repr_org_multiline(self):
        """Test _repr_org with multiline text."""
        text = "Line 1\nLine 2\nLine 3"
        o = Org(text)
        assert o._repr_org() == "Line 1\nLine 2\nLine 3\n"

    def test_str(self):
        """Test __str__ method."""
        o = Org("Test text")
        assert str(o) == o._repr_org()

    def test_repr_mimebundle(self):
        """Test _repr_mimebundle method."""
        o = Org("Test")
        bundle = o._repr_mimebundle_()
        assert "text/org" in bundle
        assert bundle["text/org"] == "Test\n"


class TestFigure:
    """Tests for the Figure class."""

    def test_basic_initialization(self):
        """Test basic figure initialization."""
        f = Figure("image.png")
        assert f.fname == "image.png"
        assert f.caption is None
        assert f.name is None
        assert f.attributes == ()

    def test_initialization_with_all_params(self):
        """Test figure with all parameters."""
        f = Figure(
            "plot.png",
            caption="A plot",
            name="fig:myplot",
            attributes=(("html", ":width 500"), ("latex", ":width 0.8\\textwidth")),
        )
        assert f.fname == "plot.png"
        assert f.caption == "A plot"
        assert f.name == "fig:myplot"
        assert len(f.attributes) == 2

    def test_repr_org_simple(self):
        """Test _repr_org for simple figure."""
        f = Figure("image.png")
        assert f._repr_org() == "[[image.png]]\n"

    def test_repr_org_with_caption(self):
        """Test _repr_org with caption."""
        f = Figure("image.png", caption="My Image")
        result = f._repr_org()
        assert "#+caption: My Image" in result
        assert "[[image.png]]" in result

    def test_repr_org_with_name(self):
        """Test _repr_org with name."""
        f = Figure("image.png", name="fig:test")
        result = f._repr_org()
        assert "#+name: fig:test" in result
        assert "[[image.png]]" in result

    def test_repr_org_with_attributes(self):
        """Test _repr_org with attributes."""
        f = Figure("image.png", attributes=(("html", ":width 500"),))
        result = f._repr_org()
        assert "#+attr_html: :width 500" in result
        assert "[[image.png]]" in result

    def test_repr_org_with_all_options(self):
        """Test _repr_org with all options."""
        f = Figure(
            "plot.png",
            caption="Test Plot",
            name="fig:test",
            attributes=(("html", ":width 500"), ("latex", ":width 0.5\\textwidth")),
        )
        result = f._repr_org()
        # Check order: attributes, then name, then caption, then link
        lines = result.strip().split("\n")
        assert lines[0].startswith("#+attr_html:")
        assert lines[1].startswith("#+attr_latex:")
        assert lines[2] == "#+name: fig:test"
        assert lines[3] == "#+caption: Test Plot"
        assert lines[4] == "[[plot.png]]"

    def test_str(self):
        """Test __str__ method."""
        f = Figure("test.png")
        assert str(f) == f._repr_org()

    def test_repr_mimebundle(self):
        """Test _repr_mimebundle method."""
        f = Figure("test.png")
        bundle = f._repr_mimebundle_()
        assert "text/org" in bundle
        assert "[[test.png]]" in bundle["text/org"]

    def test_attributes_tuple_conversion(self):
        """Test that attributes are converted to tuple."""
        f = Figure("test.png", attributes=[("html", ":width 100")])
        assert isinstance(f.attributes, tuple)

    def test_empty_attributes(self):
        """Test with empty attributes."""
        f = Figure("test.png", attributes=[])
        assert f.attributes == ()


class TestTable:
    """Tests for the Table class."""

    def test_basic_initialization(self):
        """Test basic table initialization."""
        data = [[1, 2, 3], [4, 5, 6]]
        t = Table(data)
        assert t.data == data
        assert t.headers is None
        assert t.caption is None
        assert t.name is None
        assert t.attributes == ()

    def test_initialization_with_all_params(self):
        """Test table with all parameters."""
        data = [[1, 2], [3, 4]]
        headers = ["A", "B"]
        t = Table(
            data,
            headers=headers,
            caption="Test Table",
            name="tab:test",
            attributes=(("html", ":class mytable"),),
        )
        assert t.data == data
        assert t.headers == headers
        assert t.caption == "Test Table"
        assert t.name == "tab:test"
        assert len(t.attributes) == 1

    def test_repr_org_simple(self):
        """Test _repr_org for simple table."""
        data = [[1, 2], [3, 4]]
        t = Table(data)
        result = t._repr_org()
        # Check that it contains the table structure
        assert "|" in result
        assert "1" in result
        assert "2" in result
        assert "3" in result
        assert "4" in result

    def test_repr_org_with_headers(self):
        """Test _repr_org with headers."""
        data = [[1, 2], [3, 4]]
        headers = ["Col A", "Col B"]
        t = Table(data, headers=headers)
        result = t._repr_org()
        assert "Col A" in result
        assert "Col B" in result
        # Headers should be separated by a line
        assert "|-" in result or "|---" in result

    def test_repr_org_with_caption(self):
        """Test _repr_org with caption."""
        data = [[1, 2]]
        t = Table(data, caption="My Table")
        result = t._repr_org()
        assert "#+caption: My Table" in result

    def test_repr_org_with_name(self):
        """Test _repr_org with name."""
        data = [[1, 2]]
        t = Table(data, name="tab:test")
        result = t._repr_org()
        assert "#+name: tab:test" in result

    def test_repr_org_with_attributes(self):
        """Test _repr_org with attributes."""
        data = [[1, 2]]
        t = Table(data, attributes=(("html", ":class mytable"),))
        result = t._repr_org()
        assert "#+attr_html: :class mytable" in result

    def test_repr_org_order(self):
        """Test that _repr_org has correct order."""
        data = [[1, 2]]
        t = Table(
            data,
            caption="Test",
            name="tab:test",
            attributes=(("html", ":class test"),),
        )
        result = t._repr_org()
        lines = result.strip().split("\n")
        # Order should be: attributes, name, caption, then table
        assert lines[0].startswith("#+attr_")
        assert lines[1] == "#+name: tab:test"
        assert lines[2] == "#+caption: Test"

    def test_str(self):
        """Test __str__ method."""
        data = [[1, 2]]
        t = Table(data)
        assert str(t) == t._repr_org()

    def test_repr_mimebundle(self):
        """Test _repr_mimebundle method."""
        data = [[1, 2]]
        t = Table(data)
        bundle = t._repr_mimebundle_()
        assert "text/org" in bundle
        assert "|" in bundle["text/org"]

    def test_empty_table(self):
        """Test with empty data."""
        t = Table([])
        result = t._repr_org()
        # Should still produce valid org output
        assert result.endswith("\n")

    def test_attributes_tuple_conversion(self):
        """Test that attributes are converted to tuple."""
        t = Table([[1, 2]], attributes=[("html", ":class test")])
        assert isinstance(t.attributes, tuple)


class TestOrgFormatter:
    """Tests for the OrgFormatter class."""

    def test_format_type(self):
        """Test that format_type is set correctly."""
        formatter = OrgFormatter()
        # The format_type should be a Unicode string "text/org"
        assert str(formatter.format_type) == "text/org"

    def test_print_method(self):
        """Test that print_method is set correctly."""
        formatter = OrgFormatter()
        # The print_method should be "_repr_org_"
        assert str(formatter.print_method) == "_repr_org_"


class TestRegisterOrgFormatter:
    """Tests for the _register_org_formatter function."""

    def test_register_in_ipython_context(self):
        """Test registering formatter in IPython context."""
        # This test is tricky because we need to mock get_ipython
        # We'll test that it doesn't raise an error when get_ipython exists
        mock_ip = MagicMock()
        mock_ip.display_formatter.formatters = {}

        # Import first to get the module
        from pycse import orgmode

        # Mock get_ipython in the orgmode module's globals
        with patch.dict(orgmode.__dict__, {"get_ipython": lambda: mock_ip}):
            orgmode._register_org_formatter()
            # Check that the formatter was registered
            assert "text/org" in mock_ip.display_formatter.formatters

    def test_register_outside_ipython(self):
        """Test that registration doesn't fail outside IPython."""
        # This should not raise an error
        from pycse import orgmode

        # Create a function that raises NameError when called
        def raise_name_error():
            raise NameError("name 'get_ipython' is not defined")

        # Mock get_ipython to raise NameError
        with patch.dict(orgmode.__dict__, {"get_ipython": raise_name_error}):
            # Should not raise
            orgmode._register_org_formatter()


class TestIntegration:
    """Integration tests combining multiple classes."""

    def test_complete_document(self):
        """Test creating a complete org document."""
        doc_parts = []

        # Add keyword
        doc_parts.append(str(Keyword("TITLE", "My Document")))
        doc_parts.append(str(Keyword("AUTHOR", "Test Author")))

        # Add comment
        doc_parts.append(str(Comment("This is a comment")))

        # Add heading
        doc_parts.append(str(Heading("Introduction", level=1)))

        # Add some text
        doc_parts.append(str(Org("This is some introductory text.")))

        # Add a table
        data = [[1, 2, 3], [4, 5, 6]]
        headers = ["A", "B", "C"]
        doc_parts.append(str(Table(data, headers=headers, caption="Data Table")))

        # Add a figure
        doc_parts.append(str(Figure("plot.png", caption="My Plot")))

        # Combine all parts
        doc = "".join(doc_parts)

        # Verify key elements are present
        assert "#+TITLE: My Document" in doc
        assert "#+AUTHOR: Test Author" in doc
        assert "# This is a comment" in doc
        assert "* Introduction" in doc
        assert "This is some introductory text" in doc
        assert "#+caption: Data Table" in doc
        assert "[[plot.png]]" in doc

    def test_nested_headings(self):
        """Test creating nested heading structure."""
        h1 = Heading("Top Level", level=1)
        h2 = Heading("Second Level", level=2)
        h3 = Heading("Third Level", level=3)

        assert str(h1).startswith("* ")
        assert str(h2).startswith("** ")
        assert str(h3).startswith("*** ")

    def test_table_with_various_data_types(self):
        """Test table with mixed data types."""
        data = [
            ["String", 123, 45.67],
            ["Another", 456, 78.90],
        ]
        headers = ["Text", "Integer", "Float"]
        t = Table(data, headers=headers)
        result = t._repr_org()

        # Check all values are represented
        assert "String" in result
        assert "123" in result
        assert "45.67" in result
