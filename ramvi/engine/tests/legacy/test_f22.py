import unittest
from .test_f07 import Toy,ns
class Clock:
    def __init__(self):self.t=0.
    def perf_counter(self):return self.t
class MetricsTest(unittest.TestCase):
    def test_first_token_counts_and_consumer_wait_excluded(self):
        clock=Clock();old=ns['time'];ns['time']=clock
        try:
            e=Toy(20);st={};g=e.generate([1],3,stats=st)
            self.assertEqual(next(g),7);self.assertEqual(st['gen_n'],1);self.assertEqual(st['decode_n'],0)
            clock.t+=100
            self.assertEqual(next(g),7);self.assertEqual(st['gen_ms'],0);self.assertEqual(st['gen_n'],2)
            clock.t+=100;list(g)
            self.assertEqual(st['gen_ms'],0);self.assertEqual(st['decode_n'],2)
        finally:ns['time']=old
if __name__=='__main__':unittest.main()
