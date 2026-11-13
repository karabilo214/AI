import unittest
from ..plan import measured_schedule

class ScheduleTest(unittest.TestCase):
    def test_solvable_control(self):
        rows=[dict(gpu_s=.1),dict(cpu_s=1.,gpu_s=.1,copy_s=.1)]
        route,score,base=measured_schedule(rows,['gpu','cpu'])
        self.assertEqual(route,['gpu','stream']);self.assertLess(score,base)
    def test_cpu_and_stream_extremes(self):
        for cpu,copy,want in ((.01,1.,'cpu'),(1.,.01,'stream')):
            rows=[dict(cpu_s=cpu,gpu_s=.01,copy_s=copy)]*5
            route,_,_=measured_schedule(rows,['cpu','stream','cpu','stream','cpu'])
            self.assertEqual(route,[want]*5)
    def test_batch_costs_change_route(self):
        a=[dict(cpu_s=.01,gpu_s=.1,copy_s=.1)]*3
        b=[dict(cpu_s=1.,gpu_s=.1,copy_s=.1)]*3
        self.assertNotEqual(measured_schedule(a,['stream']*3)[0],measured_schedule(b,['stream']*3)[0])
    def test_invalid_or_missing_measurements(self):
        for v in (float('nan'),float('inf'),0.,-1.,True):
            with self.assertRaises(ValueError):measured_schedule([dict(cpu_s=v)],['cpu'])
        with self.assertRaises(ValueError):measured_schedule([dict(cpu_s=.1)],['stream'])
    def test_hysteresis(self):
        route,_,_=measured_schedule([dict(cpu_s=1.,gpu_s=.98,copy_s=.001)],['cpu'])
        self.assertEqual(route,['cpu'])
    def test_gpu_residency_and_cpu_only_preserved(self):
        rows=[dict(gpu_s=2.,cpu_s=.001),dict(cpu_s=.3)]
        self.assertEqual(measured_schedule(rows,['gpu','cpu'])[0],['gpu','cpu'])

    def test_two_buffer_overlap_affects_choice(self):
        # Два первых DMA уже готовы во время CPU-слоя.
        rows=[dict(cpu_s=.3,gpu_s=.01,copy_s=.2)]*6
        initial=['cpu','stream','cpu','stream','cpu','stream']
        route,score,base=measured_schedule(rows,initial)
        self.assertLessEqual(score,base)
        self.assertEqual(len(route),6)

    def test_configure_validates_before_publishing(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from ..measured_plan import configure
        e=SimpleNamespace(L=2,place=['gpu','hyb'],_reset_loader=lambda:None,ctx=[1],graphs={1:1})
        profile=dict(version=1,signature={'test':1},place=['gpu','hyb'],batches={
            '1':dict(layers=[dict(gpu_s=.01),dict(cpu_s=1.,gpu_s=.01,copy_s=.01)],baseline_route=['gpu','cpu'])})
        with patch('llm_app.measured_plan.signature',return_value={'test':1}):
            configure(e,profile)
            self.assertEqual(e._measured_routes[1],['gpu','stream'])
            self.assertEqual(e.ctx,[])
            profile['signature']={'wrong':2}
            with self.assertRaises(ValueError):configure(e,profile)

    def test_dense_memory_plan_accepts_measured_times(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from ..plan import make_plan
        ck=SimpleNamespace(L=3,D=256,KV=1,HD=64,cfg={'vocab_size':1024},layer_params=lambda:{'qkv':(256,256)})
        costs=[dict(gpu_s=.01),dict(gpu_s=.01,copy_s=.01,cpu_s=1.),dict(gpu_s=.01,copy_s=.01,cpu_s=1.)]
        with patch('llm_app.plan.torch.cuda.mem_get_info',return_value=(2**30,2**30)),patch('llm_app.plan.ram_info',return_value=(2**30,2**30)):
            result=make_plan(ck,256,force_resident=1,layer_costs=costs,log=lambda _:None)
        self.assertEqual(result['place'],['gpu','stream','stream'])

    def test_measured_full_pass_can_veto_model(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from ..measured_plan import configure
        e=SimpleNamespace(L=1,place=['hyb'],_reset_loader=lambda:None,ctx=[],graphs={})
        profile=dict(version=1,signature={'test':1},place=['hyb'],batches={
            '1':dict(layers=[dict(cpu_s=1.,gpu_s=.01,copy_s=.01)],baseline_route=['cpu'],validated_route=['cpu'])})
        with patch('llm_app.measured_plan.signature',return_value={'test':1}):
            configure(e,profile)
            self.assertEqual(e._measured_routes[1],['cpu'])
            profile['batches']['1']['validated_route']=['gpu']
            with self.assertRaises(ValueError):configure(e,profile)

    def test_calibration_drains_prefetch_between_batches(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        import torch
        from ..measured_plan import calibrate
        class Fake:
            place=['hyb'];stream_ids=[];ctx=[];D=2;V=10;L=1;phase=None
            def __init__(self):
                self.pending=[];self.pos=torch.zeros(1);self.cbuf=[{'a':torch.zeros(1)}]
            def _reset_loader(self):self.pending=[];self.phase=None
            def _cpu_layer(self,*a):pass
            def _gpu_layer(self,*a):pass
            def _request(self,l):
                if self.pending:raise AssertionError('undrained prefetch')
                self.pending=[l]
            def _acquire(self,l):
                self.assertion=self.pending.pop(0)==l
                self.cur={'s':0,'rel':{'done':SimpleNamespace(set=lambda:None)}}
                return {}
            def _pk(self,l):return {'a':(torch.zeros(1),None)}
            def _decode(self,tokens,pos):self.pending=['ahead','ahead']
            def _pick(self,t,*a):return [7]*t
        e=Fake();ones=torch.ones
        with patch('llm_app.measured_plan.signature',return_value={}),patch('torch.cuda.synchronize'),\
             patch('torch.cuda.Event',return_value=SimpleNamespace(record=lambda:None)),\
             patch('torch.ones',side_effect=lambda shape,**kw:ones(shape,dtype=kw.get('dtype'))):
            profile=calibrate(e,batches=(1,4),repeats=3)
        self.assertEqual(set(profile['batches']),{'1','4'})
        self.assertEqual(e.pending,[])
