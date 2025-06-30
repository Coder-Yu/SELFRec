"""Shared pytest fixtures and configuration for SELFRec tests."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import pytest
import yaml


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_config_file(temp_dir: Path) -> Path:
    """Create a mock configuration file for testing."""
    config_path = temp_dir / "test_config.yaml"
    config_data = {
        "recommender": "TestModel",
        "dataset": "test_dataset",
        "training": {
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
        },
        "evaluation": {
            "metrics": ["recall", "ndcg"],
            "k": [10, 20],
        },
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path


@pytest.fixture
def mock_dataset_file(temp_dir: Path) -> Path:
    """Create a mock dataset file for testing."""
    dataset_path = temp_dir / "test_data.txt"
    with open(dataset_path, "w") as f:
        f.write("user1 item1 1\n")
        f.write("user1 item2 1\n")
        f.write("user2 item1 1\n")
        f.write("user2 item3 1\n")
    return dataset_path


@pytest.fixture
def sample_user_item_matrix():
    """Create a sample user-item interaction matrix."""
    import numpy as np
    
    matrix = np.array([
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
    ])
    return matrix


@pytest.fixture
def sample_embeddings():
    """Create sample user and item embeddings."""
    import numpy as np
    
    user_embeddings = np.random.randn(10, 64)
    item_embeddings = np.random.randn(20, 64)
    return user_embeddings, item_embeddings


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_torch_model():
    """Create a mock PyTorch model for testing."""
    import torch
    import torch.nn as nn
    
    class MockModel(nn.Module):
        def __init__(self, input_dim=100, hidden_dim=64, output_dim=10):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            return self.fc2(x)
    
    return MockModel()


@pytest.fixture
def sample_graph_data():
    """Create sample graph data for graph-based models."""
    edges = [
        (0, 1), (0, 2), (1, 2), (1, 3),
        (2, 3), (3, 4), (4, 5), (5, 0)
    ]
    num_nodes = 6
    return {"edges": edges, "num_nodes": num_nodes}


@pytest.fixture
def mock_training_config():
    """Create a mock training configuration."""
    return {
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 10,
        "device": "cpu",
        "optimizer": "adam",
        "loss": "bpr",
        "eval_interval": 5,
        "early_stop": 3,
    }


@pytest.fixture
def capture_stdout(monkeypatch):
    """Capture stdout for testing print statements."""
    import io
    import sys
    
    captured_output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured_output)
    return captured_output


@pytest.fixture
def mock_logger(mocker):
    """Create a mock logger for testing logging functionality."""
    logger = mocker.Mock()
    logger.info = mocker.Mock()
    logger.warning = mocker.Mock()
    logger.error = mocker.Mock()
    logger.debug = mocker.Mock()
    return logger


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def clean_imports():
    """Clean up imports to ensure fresh module loading."""
    import sys
    
    modules_to_remove = [
        module for module in sys.modules 
        if module.startswith(("base", "data", "model", "util"))
    ]
    
    for module in modules_to_remove:
        del sys.modules[module]
    
    yield
    
    # Clean up again after test
    modules_to_remove = [
        module for module in sys.modules 
        if module.startswith(("base", "data", "model", "util"))
    ]
    
    for module in modules_to_remove:
        del sys.modules[module]