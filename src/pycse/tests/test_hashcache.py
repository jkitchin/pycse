"""Tests for hashcache module."""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from pycse.hashcache import hashcache, HashCache, SqlCache, JsonCache


# Test deprecated hashcache function
def test_hashcache_function_deprecated():
    """Test that the old hashcache function decorator raises an exception."""
    with pytest.raises(Exception, match="deprecated"):

        @hashcache
        def f(x):
            return x


# HashCache Tests
class TestHashCache:
    """Tests for HashCache class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        original_cache = HashCache.cache
        HashCache.cache = temp_dir
        yield temp_dir
        # Cleanup
        HashCache.cache = original_cache
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    def test_basic_caching(self, temp_cache_dir):
        """Test that a function result is cached and reused."""
        call_count = 0

        @HashCache
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call - should execute function
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call with same args - should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Function not called again

    def test_different_args_different_cache(self, temp_cache_dir):
        """Test that different arguments produce different cache entries."""
        call_count = 0

        @HashCache
        def add(a, b):
            nonlocal call_count
            call_count += 1
            return a + b

        result1 = add(1, 2)
        result2 = add(3, 4)
        result3 = add(1, 2)  # Same as first call

        assert result1 == 3
        assert result2 == 7
        assert result3 == 3
        assert call_count == 2  # Only called twice (result3 was cached)

    def test_kwargs_caching(self, temp_cache_dir):
        """Test caching with keyword arguments."""
        call_count = 0

        @HashCache
        def func_with_defaults(x, y=10):
            nonlocal call_count
            call_count += 1
            return x + y

        # Call with default
        result1 = func_with_defaults(5)
        assert result1 == 15
        assert call_count == 1

        # Call explicitly with same default value
        result2 = func_with_defaults(5, y=10)
        assert result2 == 15
        assert call_count == 1  # Should use cache (standardized args)

        # Call with different kwarg value
        result3 = func_with_defaults(5, y=20)
        assert result3 == 25
        assert call_count == 2

    def test_get_standardized_args(self, temp_cache_dir):
        """Test that standardized args include defaults."""

        @HashCache
        def func(a, b=5, c=10):
            return a + b + c

        # Get the decorator instance
        std_args = func.get_standardized_args((1,), {})
        assert std_args == {"a": 1, "b": 5, "c": 10}

        std_args2 = func.get_standardized_args((1, 2), {"c": 20})
        assert std_args2 == {"a": 1, "b": 2, "c": 20}

    def test_get_hash(self, temp_cache_dir):
        """Test that hash generation works."""

        @HashCache
        def simple(x):
            return x

        hash1 = simple.get_hash((5,), {})
        hash2 = simple.get_hash((5,), {})
        hash3 = simple.get_hash((6,), {})

        # Same args should produce same hash
        assert hash1 == hash2
        # Different args should produce different hash
        assert hash1 != hash3

    def test_cache_file_created(self, temp_cache_dir):
        """Test that cache files are created in correct location."""

        @HashCache
        def cached_func(x):
            return x**2

        assert cached_func(3) == 9

        # Check that cache directory has files
        cache_path = Path(temp_cache_dir)
        assert cache_path.exists()
        # Should have subdirectories (first 2 chars of hash)
        subdirs = list(cache_path.iterdir())
        assert len(subdirs) > 0

    def test_verbose_mode(self, temp_cache_dir, capsys):
        """Test verbose mode prints cache information."""
        HashCache.verbose = True

        @HashCache
        def verbose_func(x):
            return x * 3

        # First call - should print when dumping
        verbose_func(4)
        captured = capsys.readouterr()
        assert "wrote" in captured.out

        # Second call - should print when loading
        verbose_func(4)
        # Should print the cached data

        HashCache.verbose = False  # Reset

    def test_mutable_argument_warning(self, temp_cache_dir, capsys):
        """Test warning for mutable arguments."""

        @HashCache
        def mutating_func(lst):
            lst.append(1)
            return lst

        my_list = [1, 2, 3]
        mutating_func(my_list)

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "mutated" in captured.out

    def test_hashcache_dump_load(self, temp_cache_dir):
        """Test HashCache static dump and load methods."""
        # Dump some data
        hsh = HashCache.dump(x=10, y=20, cache=temp_cache_dir)
        assert isinstance(hsh, str)
        assert len(hsh) > 0

        # Load it back
        loaded_data = HashCache.load(hsh, cache=temp_cache_dir)
        assert loaded_data == {"x": 10, "y": 20}

    def test_hashcache_load_nonexistent(self, temp_cache_dir):
        """Test loading non-existent hash returns None."""
        result = HashCache.load("nonexistent_hash", cache=temp_cache_dir)
        assert result is None


# SqlCache Tests
class TestSqlCache:
    """Tests for SqlCache class."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary sqlite database for testing."""
        temp_file = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        temp_file.close()
        original_cache = SqlCache.cache
        SqlCache.cache = temp_file.name
        yield temp_file.name
        # Cleanup
        SqlCache.cache = original_cache
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

    def test_sqlcache_basic_caching(self, temp_db_path):
        """Test SqlCache basic caching functionality."""
        call_count = 0

        @SqlCache
        def sql_cached_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call
        result1 = sql_cached_func(5)
        assert result1 == 10
        assert call_count == 1

        # Second call - should use cache
        result2 = sql_cached_func(5)
        assert result2 == 10
        assert call_count == 1

    def test_sqlcache_with_numpy(self, temp_db_path):
        """Test SqlCache with numpy arrays (tests orjson serialization)."""
        import numpy as np

        @SqlCache
        def numpy_func(arr):
            return arr * 2

        arr = np.array([1, 2, 3])
        result1 = numpy_func(arr)
        result2 = numpy_func(arr)  # Should use cache

        np.testing.assert_array_equal(result1, np.array([2, 4, 6]))
        np.testing.assert_array_equal(result2, np.array([2, 4, 6]))

    def test_sqlcache_search(self, temp_db_path):
        """Test SqlCache search functionality."""

        @SqlCache
        def searchable_func(x):
            return x**2

        # Create some cache entries
        searchable_func(2)
        searchable_func(3)

        # Search for all entries
        results = SqlCache.search("SELECT * FROM cache")
        rows = list(results.fetchall())
        assert len(rows) >= 2

    def test_sqlcache_dump_load(self, temp_db_path):
        """Test SqlCache dump and load methods."""
        # Dump data
        hsh = SqlCache.dump(a=100, b=200)
        assert isinstance(hsh, str)

        # Load it back
        loaded = SqlCache.load(hsh)
        assert loaded == {"a": 100, "b": 200}

    def test_sqlcache_duplicate_dump(self, temp_db_path):
        """Test that dumping duplicate data doesn't fail."""
        # Dump same data twice
        hsh1 = SqlCache.dump(test=123)
        hsh2 = SqlCache.dump(test=123)

        # Should return same hash and not raise error
        assert hsh1 == hsh2


# JsonCache Tests
class TestJsonCache:
    """Tests for JsonCache class."""

    @pytest.fixture
    def temp_json_cache_dir(self):
        """Create a temporary directory for JSON cache."""
        temp_dir = tempfile.mkdtemp()
        original_cache = JsonCache.cache
        JsonCache.cache = Path(temp_dir)
        yield temp_dir
        # Cleanup
        JsonCache.cache = original_cache
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    def test_jsoncache_basic_caching(self, temp_json_cache_dir):
        """Test JsonCache basic caching functionality."""
        call_count = 0

        @JsonCache
        def json_cached_func(x):
            nonlocal call_count
            call_count += 1
            return x * 3

        # First call
        result1 = json_cached_func(4)
        assert result1 == 12
        assert call_count == 1

        # Second call - should use cache
        result2 = json_cached_func(4)
        assert result2 == 12
        assert call_count == 1

    def test_jsoncache_with_numpy(self, temp_json_cache_dir):
        """Test JsonCache with numpy arrays."""
        import numpy as np

        @JsonCache
        def numpy_json_func(arr):
            return arr + 10

        arr = np.array([1, 2, 3])
        result1 = numpy_json_func(arr)
        result2 = numpy_json_func(arr)  # Should use cache

        np.testing.assert_array_equal(result1, np.array([11, 12, 13]))
        np.testing.assert_array_equal(result2, np.array([11, 12, 13]))

    def test_jsoncache_file_created(self, temp_json_cache_dir):
        """Test that JSON cache files are created."""

        @JsonCache
        def creates_json(x):
            return x

        assert creates_json(42) == 42

        # Check that JSON files exist
        cache_path = Path(temp_json_cache_dir)
        json_files = list(cache_path.rglob("*.json"))
        # Should have at least the Filestore.json and cached data
        assert len(json_files) >= 2

    def test_jsoncache_dump_load(self, temp_json_cache_dir):
        """Test JsonCache dump and load methods."""
        # Dump data
        hsh = JsonCache.dump(foo="bar", num=42)
        assert isinstance(hsh, str)

        # Load it back
        loaded = JsonCache.load(hsh)
        assert loaded == {"foo": "bar", "num": 42}

    def test_jsoncache_load_nonexistent(self, temp_json_cache_dir):
        """Test loading non-existent hash returns None."""
        result = JsonCache.load("nonexistent_hash")
        assert result is None

    def test_jsoncache_verbose(self, temp_json_cache_dir, capsys):
        """Test JsonCache verbose mode."""
        JsonCache.verbose = True

        @JsonCache
        def verbose_json_func(x):
            return x * 2

        # First call
        verbose_json_func(5)

        # Second call should print cached data
        verbose_json_func(5)
        # capsys will capture output
        capsys.readouterr()

        JsonCache.verbose = False  # Reset
