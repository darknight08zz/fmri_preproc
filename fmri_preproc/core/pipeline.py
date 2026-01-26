from typing import List, Any

class Pipeline:
    """
    Linear pipeline executor.
    """
    def __init__(self):
        self.nodes = []

    def add_node(self, node: Any):
        self.nodes.append(node)

    def run(self, context):
        """
        Runs all nodes in order.
        """
        print("Starting Pipeline execution...")
        for node in self.nodes:
            node_name = node.__class__.__name__
            print(f"--- Running {node_name} ---")
            # This is a simplification. Real pipeline needs data passing logic.
            # Assuming nodes know how to fetch from context or previous output.
            # For now, we will expect nodes to have a generic 'run' or we wrap them.
            if hasattr(node, 'execute'):
                 node.execute(context)
            else:
                 print(f"Node {node_name} has no execute method, skipping.")
        print("Pipeline execution finished.")
