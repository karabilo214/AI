import unittest
from ramvi.engine.tools.bench.bench_validation import paired_rows,validate_speed,compare_run
class ComparisonTest(unittest.TestCase):
    def test_mismatched_or_empty_measurements_rejected(self):
        for a,b in (([],[]),([{'q':'a'}],[{'q':'b'}]),([{'q':'a'}],[])):
            with self.assertRaises(ValueError):paired_rows(a,b,'q')
        for row in ({'n':0,'gen_s':0,'tps':1,'out':'x'},{'n':1,'gen_s':1,'tps':1,'out':''}):
            with self.assertRaises(ValueError):validate_speed([row])
    def test_generation_settings_and_weights_must_match(self):
        a={'version':1,'model':'test','parameters':{'temperature':0},'server':{'weights_id':'same'}}
        compare_run(a,a)
        for other in (None,{**a,'parameters':{'temperature':1}}, {**a,'server':{'weights_id':'other'}}):
            with self.assertRaises(ValueError):compare_run(a,other)
if __name__=='__main__':unittest.main()
