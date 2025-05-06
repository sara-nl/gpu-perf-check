"""
Base classes for GPU benchmarking.

This module provides abstract base classes and interfaces for all benchmark tests.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import torch
import json
import os
from datetime import datetime


class BenchmarkResult:
    """
    Container for benchmark results with serialization capabilities.
    """
    def __init__(
        self,
        name: str,
        device_info: Dict[str, Any],
        parameters: Dict[str, Any],
        results: Dict[str, Any]
    ):
        """
        Initialize a benchmark result.
        
        Args:
            name: Name of the benchmark
            device_info: Information about the device used
            parameters: Parameters used for the benchmark
            results: Raw results data
        """
        self.name = name
        self.device_info = device_info
        self.parameters = parameters
        self.results = results
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            'name': self.name,
            'device_info': self.device_info,
            'parameters': self.parameters,
            'results': self.results,
            'timestamp': self.timestamp
        }
    
    def save_json(self, filename: str) -> None:
        """
        Save the result to a JSON file.
        
        Args:
            filename: Path to save the JSON file
        """
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Results saved to {filename}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """
        Create a BenchmarkResult from a dictionary.
        
        Args:
            data: Dictionary with benchmark result data
            
        Returns:
            BenchmarkResult instance
        """
        result = cls(
            name=data['name'],
            device_info=data['device_info'],
            parameters=data['parameters'],
            results=data['results']
        )
        result.timestamp = data.get('timestamp', datetime.now().isoformat())
        return result
    
    @classmethod
    def load_json(cls, filename: str) -> 'BenchmarkResult':
        """
        Load a BenchmarkResult from a JSON file.
        
        Args:
            filename: Path to the JSON file
            
        Returns:
            BenchmarkResult instance
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class BenchmarkTest(ABC):
    """
    Abstract base class for all benchmark tests.
    """
    def __init__(
        self,
        name: str,
        device_id: int = 0,
        output_dir: Optional[str] = None
    ):
        """
        Initialize a benchmark test.
        
        Args:
            name: Name of the benchmark
            device_id: CUDA device ID to use
            output_dir: Directory to save results
        """
        self.name = name
        self.device_id = device_id
        self.output_dir = output_dir or os.getcwd()
        
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set the device
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            self.device = torch.device(f"cuda:{device_id}")
        else:
            raise RuntimeError("CUDA is not available")
    
    @abstractmethod
    def run(self, *args, **kwargs) -> BenchmarkResult:
        """
        Run the benchmark and return results.
        
        Returns:
            BenchmarkResult containing the benchmark results
        """
        pass
    
    @abstractmethod
    def plot(self, result: BenchmarkResult, *args, **kwargs) -> None:
        """
        Plot the benchmark results.
        
        Args:
            result: BenchmarkResult to plot
        """
        pass
    
    def save_results(self, result: BenchmarkResult, filename: Optional[str] = None) -> str:
        """
        Save benchmark results to a file.
        
        Args:
            result: BenchmarkResult to save
            filename: Optional filename to use
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            device_name = result.device_info.get('name', 'unknown').replace(' ', '-')
            filename = f"{self.name}_{device_name}_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        result.save_json(filepath)
        return filepath


class TestFunction:
    """
    Wrapper for a test function with metadata.
    """
    def __init__(
        self,
        name: str,
        function: callable,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a test function.
        
        Args:
            name: Name of the test function
            function: The function to call
            description: Optional description
            parameters: Optional parameters for the function
        """
        self.name = name
        self.function = function
        self.description = description or ""
        self.parameters = parameters or {}
    
    def __call__(self, *args, **kwargs):
        """Call the wrapped function."""
        return self.function(*args, **kwargs)


class TestFunctionRegistry:
    """
    Registry for test functions.
    """
    def __init__(self):
        """Initialize an empty registry."""
        self._functions = {}
    
    def register(
        self,
        name: str,
        function: callable,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a test function.
        
        Args:
            name: Name of the test function
            function: The function to register
            description: Optional description
            parameters: Optional parameters for the function
        """
        self._functions[name] = TestFunction(
            name=name,
            function=function,
            description=description,
            parameters=parameters
        )
    
    def get(self, name: str) -> TestFunction:
        """
        Get a test function by name.
        
        Args:
            name: Name of the test function
            
        Returns:
            TestFunction instance
        """
        if name not in self._functions:
            raise ValueError(f"Test function '{name}' not found")
        return self._functions[name]
    
    def list(self) -> List[str]:
        """
        List all registered test functions.
        
        Returns:
            List of test function names
        """
        return list(self._functions.keys())
    
    def __contains__(self, name: str) -> bool:
        """Check if a test function is registered."""
        return name in self._functions
