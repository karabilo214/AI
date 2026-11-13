import unittest
from .audit_support import definition
class DraftSizeTest(unittest.TestCase):
    def test_invalid_k_rejected_before_computing(self):
        cls=definition('engine.py','Engine',{})
        for k in (-1,0,16,32,True,1.5):
            with self.subTest(k=k),self.assertRaises(ValueError):
                next(object.__new__(cls).generate([1],k=k))
    def test_grammar_path_rejected_before_matcher_access(self):
        cls=definition('constrain.py','Constraint',{})
        with self.assertRaises(ValueError):object.__new__(cls).masks([1]*16)
if __name__=='__main__':unittest.main()
