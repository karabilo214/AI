import io,types,unittest
from llm_app.server import App,make_handler
from llm_app.request_validation import validate_request
class InputTest(unittest.TestCase):
    def test_invalid_http_bodies_and_lengths(self):
        app=App(None,None,'test');H=make_handler(app)
        for body,length,expected in [(b'[]','2',400),(b'{}','bad',400),(b'{}','2097153',413),(b'\xff','1',400),(b'{}','3',400)]:
            h=object.__new__(H);h.headers={'Content-Length':length};h.rfile=io.BytesIO(body);h.wfile=io.BytesIO();h.path='/v1/chat/completions'
            codes=[];h.send_response=lambda c:codes.append(c);h.send_header=lambda *a:None;h.end_headers=lambda:None
            h.do_POST();self.assertEqual(codes,[expected])
    def test_numeric_and_nested_validation(self):
        for req in [{'temperature':float('nan')},{'top_p':0},{'options':[]},{'messages':[1]}, {'max_tokens':-1},{'tools':[1]}]:
            with self.assertRaises(ValueError):validate_request(req)
        self.assertEqual(validate_request({'temperature':0,'max_tokens':0})['max_tokens'],0)
    def test_queue_is_bounded(self):
        app=App(None,None,'test');held=[]
        for _ in range(8):self.assertTrue(app.slots.acquire(False));held.append(1)
        self.assertFalse(app.slots.acquire(False))
        for _ in held:app.slots.release()
if __name__=='__main__':unittest.main()
