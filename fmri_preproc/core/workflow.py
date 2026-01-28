
from typing import List, Dict, Tuple, Set, Any
from collections import deque
from fmri_preproc.core.node import Node

class Workflow:
    """
    Manages a directed acyclic graph (DAG) of processing nodes.
    """
    def __init__(self, name: str):
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Tuple[str, str, str, str]] = [] # (source_node, source_out, dest_node, dest_in)
        self.adjacency: Dict[str, List[str]] = {} # source_name -> [dest_names]

    def add_node(self, node: Node):
        """Adds a node to the workflow."""
        if node.name in self.nodes:
            raise ValueError(f"Node with name '{node.name}' already exists.")
        self.nodes[node.name] = node
        self.adjacency[node.name] = []

    def connect(self, source_node: Node, source_output: str, dest_node: Node, dest_input: str):
        """Connects output of one node to input of another."""
        if source_node.name not in self.nodes:
            self.add_node(source_node)
        if dest_node.name not in self.nodes:
            self.add_node(dest_node)
            
        self.edges.append((source_node.name, source_output, dest_node.name, dest_input))
        self.adjacency[source_node.name].append(dest_node.name)

    def _get_topological_sort(self) -> List[Node]:
        """Returns nodes in execution order using Kahn's algorithm."""
        in_degree = {name: 0 for name in self.nodes}
        for u in self.adjacency:
            for v in self.adjacency[u]:
                in_degree[v] += 1

        queue = deque([name for name in in_degree if in_degree[name] == 0])
        sorted_nodes = []

        while queue:
            u_name = queue.popleft()
            sorted_nodes.append(self.nodes[u_name])

            for v_name in self.adjacency[u_name]:
                in_degree[v_name] -= 1
                if in_degree[v_name] == 0:
                    queue.append(v_name)

        if len(sorted_nodes) != len(self.nodes):
            raise RuntimeError("Cycle detected in workflow! Pipeline cannot be circular.")
            
        return sorted_nodes

    def run(self, global_context: Dict[str, Any] = None, status_callback=None):
        """
        Executes the pipeline.
        
        Args:
            global_context: Shared context (optional).
            status_callback: Function(node_name: str, status: str) to report progress.
        """
        if global_context is None:
            global_context = {}
            
        execution_order = self._get_topological_sort()
        print(f"Workflow '{self.name}' Execution Order: {[n.name for n in execution_order]}")
        
        for node in execution_order:
            # Report Start
            if status_callback:
                status_callback(node.name, "running")
            
            # 1. Resolve inputs from dependencies
            # Find edges pointing TO this node
            relevant_edges = [e for e in self.edges if e[2] == node.name]
            for src_name, src_out, dst_name, dst_in in relevant_edges:
                src_node = self.nodes[src_name]
                val = src_node.get_output(src_out)
                if val is None:
                    raise RuntimeError(f"Node '{src_name}' produced None for output '{src_out}', required by '{dst_name}'.")
                node.set_input(dst_in, val)
            
            # 2. Run Node
            node.run(global_context)
            
            # Report Completion
            if status_callback:
                status_callback(node.name, "completed")
