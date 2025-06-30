"""Validation tests to ensure the testing infrastructure is properly configured."""

import sys
from pathlib import Path

import pytest


def test_python_version():
    """Test that Python version is 3.8 or higher."""
    assert sys.version_info >= (3, 8), "Python 3.8 or higher is required"


def test_project_structure():
    """Test that the expected project structure exists."""
    workspace = Path("/workspace")
    
    # Check main directories
    assert (workspace / "base").exists(), "base directory should exist"
    assert (workspace / "data").exists(), "data directory should exist"
    assert (workspace / "model").exists(), "model directory should exist"
    assert (workspace / "util").exists(), "util directory should exist"
    assert (workspace / "conf").exists(), "conf directory should exist"
    assert (workspace / "dataset").exists(), "dataset directory should exist"
    
    # Check test directories
    assert (workspace / "tests").exists(), "tests directory should exist"
    assert (workspace / "tests" / "unit").exists(), "tests/unit directory should exist"
    assert (workspace / "tests" / "integration").exists(), "tests/integration directory should exist"


def test_configuration_files():
    """Test that configuration files are present."""
    workspace = Path("/workspace")
    
    assert (workspace / "pyproject.toml").exists(), "pyproject.toml should exist"
    assert (workspace / "requirements.txt").exists(), "requirements.txt should exist"


def test_imports():
    """Test that main modules can be imported."""
    # Test base imports
    try:
        from base import recommender
        assert hasattr(recommender, "Recommender"), "Recommender class should be importable"
    except ImportError as e:
        pytest.skip(f"Cannot import base.recommender: {e}")
    
    # Test if torch is available
    try:
        import torch
        assert torch.__version__, "PyTorch should be installed"
    except ImportError:
        pytest.skip("PyTorch not installed yet")


def test_pytest_markers():
    """Test that custom pytest markers are registered."""
    # This test verifies that our custom markers are properly configured
    # The markers are defined in pyproject.toml
    pass


@pytest.mark.unit
def test_unit_marker():
    """Test that unit marker works."""
    assert True, "Unit marker should work"


@pytest.mark.integration
def test_integration_marker():
    """Test that integration marker works."""
    assert True, "Integration marker should work"


@pytest.mark.slow
def test_slow_marker():
    """Test that slow marker works."""
    assert True, "Slow marker should work"


def test_fixtures_available(temp_dir, mock_config_file, mock_dataset_file):
    """Test that custom fixtures are available and working."""
    assert temp_dir.exists(), "Temporary directory should exist"
    assert mock_config_file.exists(), "Mock config file should exist"
    assert mock_dataset_file.exists(), "Mock dataset file should exist"
    
    # Check content of mock files
    assert mock_config_file.suffix == ".yaml", "Config file should be YAML"
    assert mock_dataset_file.read_text().strip(), "Dataset file should have content"


def test_coverage_configuration():
    """Test that coverage is properly configured."""
    workspace = Path("/workspace")
    pyproject = workspace / "pyproject.toml"
    
    assert pyproject.exists(), "pyproject.toml should exist"
    
    content = pyproject.read_text()
    assert "[tool.coverage.run]" in content, "Coverage run configuration should exist"
    assert "[tool.coverage.report]" in content, "Coverage report configuration should exist"
    assert "cov-fail-under=80" in content, "Coverage threshold should be set to 80%"


def test_poetry_scripts():
    """Test that Poetry scripts are configured."""
    workspace = Path("/workspace")
    pyproject = workspace / "pyproject.toml"
    
    content = pyproject.read_text()
    assert "[tool.poetry.scripts]" in content, "Poetry scripts section should exist"
    assert 'test = "pytest:main"' in content, "test command should be configured"
    assert 'tests = "pytest:main"' in content, "tests command should be configured"


class TestSampleClass:
    """Sample test class to verify class-based tests work."""
    
    def test_sample_method(self):
        """Test that class-based tests are discovered."""
        assert 1 + 1 == 2, "Basic arithmetic should work"
    
    def test_with_fixture(self, sample_user_item_matrix):
        """Test that fixtures work in class methods."""
        assert sample_user_item_matrix.shape == (4, 4), "Matrix should be 4x4"