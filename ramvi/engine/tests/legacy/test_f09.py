import contextlib,queue,threading,time,types,unittest
from .audit_support import definition
class Event:
    def record(self,*a):pass
    def query(self):return True
    def synchronize(self):pass
class Stream:
    device=0
    def wait_event(self,*a):pass
    def query(self):return True
class Buffer:
    def __init__(self,fail=False):self.fail=fail
    def numel(self):return 1
    def __getitem__(self,key):return self
    def copy_(self,*a,**kw):
        if self.fail:raise ValueError('copy failed')
class LoaderTest(unittest.TestCase):
    def test_success_error_and_shutdown(self):
        cuda=types.SimpleNamespace(Stream=Stream,Event=Event,set_device=lambda *a:None,stream=lambda s:contextlib.nullcontext())
        cls=definition('engine.py','Loader',dict(torch=types.SimpleNamespace(cuda=cuda),queue=queue,threading=threading,time=time))
        for fail in (False,True):
            loader=cls()
            job=dict(wait=None,pairs=[(Buffer(),Buffer(fail))],done=threading.Event(),rel={'done':threading.Event()})
            try:
                loader.q.put(job)
                if fail:
                    with self.assertRaises(RuntimeError) as e:loader.wait(job)
                    self.assertIsInstance(e.exception.__cause__,ValueError)
                else:loader.wait(job);self.assertIsInstance(job['ev'],Event)
            finally:loader.close()
            self.assertFalse(loader.thread.is_alive())
    def test_wait_has_timeout(self):
        cls=definition('engine.py','Loader',dict(time=time))
        cls.TIMEOUT=.001
        with self.assertRaises(TimeoutError):cls.wait({'done':threading.Event()})
if __name__=='__main__':unittest.main()
