
from typing import Dict, Any, List, Optional
import hashlib
import json
import os
from abc import ABC, abstractmethod

class Node(ABC):
    """
    Base class for a processing node in the pipeline.
    """
    def __init__(self, name: str):
        self.name = name
        self.inputs: Dict[str, Any] = {}
        self.outputs: Dict[str, Any] = {}
        self.required_inputs: List[str] = []
        
    def set_input(self, key: str, value: Any):
        """Sets a specific input value."""
        if key not in self.required_inputs:
             # Flexible inputs are allowed, but we warn/log if needed in strict mode
             pass
        self.inputs[key] = value

    def get_output(self, key: str) -> Any:
        """Retrieves a specific output value."""
        return self.outputs.get(key)

    def validate(self) -> bool:
        """Checks if all required inputs are present."""
        missing = [key for key in self.required_inputs if key not in self.inputs]
        if missing:
            raise ValueError(f"Node '{self.name}' missing required inputs: {missing}")
        return True

    @abstractmethod
    def execute(self, context: Dict[str, Any]):
        """
        Execute the node logic. 
        Must populate self.outputs.
        """
        pass

    def __repr__(self):
        return f"<Node: {self.name}>"

    def run(self, context: Dict[str, Any] = None):
        """Wrapper for execute that handles validation and logging."""
        print(f"[{self.name}] Validating...")
        self.validate()
        print(f"[{self.name}] Executing...")
        if context is None:
            context = {}
        self.execute(context)
        print(f"[{self.name}] Finished.")
