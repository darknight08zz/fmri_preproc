
import unittest
from fmri_preproc.core.node import Node
from fmri_preproc.core.workflow import Workflow

class AddNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.required_inputs = ['a', 'b']
        
    def execute(self, context):
        res = self.inputs['a'] + self.inputs['b']
        self.outputs['sum'] = res
        print(f"[{self.name}] {self.inputs['a']} + {self.inputs['b']} = {res}")

class MultiplyNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.required_inputs = ['x', 'y']
        
    def execute(self, context):
        res = self.inputs['x'] * self.inputs['y']
        self.outputs['product'] = res
        print(f"[{self.name}] {self.inputs['x']} * {self.inputs['y']} = {res}")

class TestWorkflow(unittest.TestCase):
    def test_simple_dag(self):
        # A = 5 + 3 = 8
        # B = A * 2 = 16
        
        node1 = AddNode("AddStep")
        node1.set_input('a', 5)
        node1.set_input('b', 3)
        
        node2 = MultiplyNode("MultStep")
        node2.set_input('y', 2) # Constant input
        
        wf = Workflow("MathTest")
        wf.connect(node1, 'sum', node2, 'x')
        
        wf.run()
        
        self.assertEqual(node1.get_output('sum'), 8)
        self.assertEqual(node2.get_output('product'), 16)

if __name__ == '__main__':
    unittest.main()
