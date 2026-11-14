"""O02: выбор до draft, ограниченная пауза и возврат к угадыванию."""
import types
import unittest
from unittest.mock import patch
import numpy as np
from .audit_support import definition

class Clock:
    def __init__(self): self.now = 0.
    def perf_counter(self): return self.now
clock = Clock()
Engine = definition('engine.py', 'Engine', dict(np=np, time=clock,
    torch=types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None))))

class Toy(Engine):
    def __init__(self, good=True):
        self.maxt=4096; self.ctx=[]; self.ck=types.SimpleNamespace(eos={999})
        self.good=good; self.dcalls=0
    def process(self, tokens, pos0, *args, **kwargs):
        clock.now += .01 + .001 * len(tokens)
        self.ctx=self.ctx[:pos0]+list(tokens)
        return [7]*len(tokens)
    def draft(self, seq, k, cum=None):
        self.dcalls += 1; clock.now += .005*k
        self.draft_confs=[1.]*k
        return [7 if self.good else 8]*k

class GateTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict('os.environ', {'LLM_DRAFT_GATE': '1'})
        env.start(); self.addCleanup(env.stop)

    def test_forced_step_never_calls_draft(self):
        e,d=Toy(),Toy(); st={}
        self.assertEqual(list(e.generate([1],90,draft=d,force=lambda _: 'step',stats=st)),[7]*90)
        self.assertEqual(d.dcalls,0); self.assertEqual(st['draft_calls'],0)
    def test_bad_draft_is_skipped_but_probed(self):
        e,d=Toy(),Toy(False); st={}
        self.assertEqual(list(e.generate([1],100,draft=d,stats=st)),[7]*100)
        self.assertGreater(st['draft_skips'],50)
        self.assertGreater(st['draft_probes'],2)
        self.assertLess(d.dcalls,30)
    def test_good_draft_and_fixed_k(self):
        for k in (None,3,15):
            e,d=Toy(),Toy(); st={}
            self.assertEqual(list(e.generate([1,2],100,draft=d,k=k,stats=st)),[7]*100)
            self.assertGreater(st['accepted'],50)
    def test_recovery_and_maximum_probe_gap(self):
        e,d=Toy(),Toy(False)
        g=e.generate([1],400,draft=d)
        for _ in range(100): next(g)
        before=d.dcalls; d.good=True
        for _ in range(100): next(g)
        self.assertGreater(d.dcalls,before+3)
        self.assertEqual(e._cost['last'],'spec')
        g.close()
        e._cost.update(gate_key=(id(d),()),gate_samples=3,gate_s=1.,gate_yield=1.,c1=.01,last='step',gate_steps=8)
        self.assertEqual(e._draft_plan(1,15,d),'probe')
    def test_new_request_probes_and_new_draft_resets(self):
        e,d=Toy(),Toy(False)
        list(e.generate([1],50,draft=d))
        self.assertEqual(e._draft_plan(0,15,d),'probe')
        self.assertEqual(e._draft_plan(1,15,Toy()),'spec')
        self.assertEqual(e._draft_plan(1,0,d),'step')

    def test_eos_sampling_and_tiny_budget(self):
        e,d=Toy(),Toy(); e.ck.eos={7}
        self.assertEqual(list(e.generate([1],100,draft=d)),[7])
        self.assertEqual(d.dcalls,0)
        e=Toy()
        self.assertEqual(list(e.generate([1],20,temperature=.7,draft=d)),[7]*20)
        self.assertEqual(d.dcalls,0)
        for count in (0,1,2):
            self.assertEqual(list(Toy().generate([1],count,draft=d)),[7]*count)
    def test_previous_acceptance_wins_over_confidence(self):
        e,d=Toy(),Toy()
        e._cost=dict(gate_key=(id(d),()),gate_samples=3,gate_s=1.,
                     gate_yield=1.,c1=.01,last='spec')
        d.draft_confs=[1.]*15
        self.assertEqual(e._draft_plan(5,15,d),'step')

    def test_cpu_decision_accounts_for_probe_cost(self):
        e,d=Toy(),Toy()
        e._cost=dict(gate_key=(id(d),()),gate_samples=3,gate_s=.4,
                     gate_yield=2.,c1=.17,last='spec')
        self.assertEqual(e._draft_plan(3,15,d),'spec')

    def test_fresh_request_replaces_old_acceptance_estimate(self):
        e,d=Toy(),Toy()
        e._cost=dict(td=None,pts=[],gate_key=(id(d),()),gate_samples=30,
                     gate_s=1.,gate_yield=1.,c1=.011,last='step')
        self.assertEqual(list(e.generate([1],17,draft=d)),[7]*17)
        self.assertEqual(e._cost['gate_yield'],16.)


class DefaultPolicyTest(unittest.TestCase):
    def test_default_and_explicit_zero_use_legacy_policy(self):
        for value in (None, '0'):
            with patch.dict('os.environ', {}, clear=True):
                if value is not None:
                    __import__('os').environ['LLM_DRAFT_GATE'] = value
                e,d=Toy(),Toy(False);st={};decisions=[]
                def decide(draft,n):
                    self.assertGreater(draft.dcalls,len(decisions))
                    decisions.append(n)
                    return True
                e._prefer_step=decide
                e._draft_plan=lambda *a: self.fail('experimental gate called by default')
                self.assertEqual(list(e.generate([1],25,draft=d,stats=st)),[7]*25)
                self.assertTrue(decisions)
                self.assertEqual(st['draft_probes'],0)
                self.assertNotIn('gate_samples',e._cost)
    def test_explicit_one_uses_predecision(self):
        with patch.dict('os.environ',{'LLM_DRAFT_GATE':'1'}):
            e,d=Toy(),Toy(False);st={}
            e._prefer_step=lambda *a:self.fail('legacy policy used with opt-in')
            self.assertEqual(list(e.generate([1],100,draft=d,stats=st)),[7]*100)
            self.assertGreater(st['draft_skips'],50)
            self.assertGreater(st['draft_probes'],0)
