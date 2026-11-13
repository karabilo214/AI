import unittest
from .test_f07 import Toy
class BudgetTest(unittest.TestCase):
    def test_zero_does_no_compute(self):
        e=Toy(8);st={};self.assertEqual(list(e.generate([],0,stats=st)),[])
        self.assertEqual(e.calls,[]);self.assertEqual(st['gen_n'],0)
    def test_invalid_budget_and_empty_prompt(self):
        for value in (-1,True,1.5):
            with self.assertRaises(ValueError):list(Toy(8).generate([1],value))
        with self.assertRaisesRegex(ValueError,'пустой'):list(Toy(8).generate([],1))
if __name__=='__main__':unittest.main()
