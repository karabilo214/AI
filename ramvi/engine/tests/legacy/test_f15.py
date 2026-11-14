import io,json,threading,time,types,unittest
from ramvi_engine.server.app import App,make_handler
from .test_f12 import Tk,Eng
class StreamErrorTest(unittest.TestCase):
    def test_stream_error_never_starts_second_http_response(self):
        app=App(Eng(1),Tk(['a']),'test')
        app.build_ids=lambda *a:[1];app.constraint=lambda *a:None
        def fail(*a,**k):
            yield 'text','hello'
            raise RuntimeError('compute failed')
        app.run=fail
        H=make_handler(app);h=object.__new__(H)
        body=json.dumps({'stream':True}).encode();h.headers={'Content-Length':str(len(body))}
        h.path='/v1/chat/completions';h.rfile=io.BytesIO(body);h.wfile=io.BytesIO();codes=[]
        h.send_response=lambda c:codes.append(c);h.send_header=lambda *a:None;h.end_headers=lambda:None
        h.do_POST()
        self.assertEqual(codes,[200]);self.assertIn(b'compute failed',h.wfile.getvalue());self.assertIn(b'[DONE]',h.wfile.getvalue())
    def test_cancellation_releases_model_worker(self):
        app=App(Eng(10000),Tk(['a']*10000),'test')
        gen=app.run([1]);next(gen);gen.close()
        self.assertFalse(any(t.name=='llm-response' and t.is_alive() for t in threading.enumerate()))
    def test_slow_consumer_does_not_wait_for_dead_producer(self):
        app=App(Eng(1000),Tk(['a']*1000),'test')
        app.QUEUE_SIZE=1;app.QUEUE_TIMEOUT=.01
        gen=app.run([1]);next(gen)
        time.sleep(.06)
        started=time.monotonic()
        with self.assertRaisesRegex(TimeoutError,'потребитель'):
            list(gen)
        self.assertLess(time.monotonic()-started,1)
        self.assertFalse(any(t.name=='llm-response' and t.is_alive() for t in threading.enumerate()))
if __name__=='__main__':unittest.main()
