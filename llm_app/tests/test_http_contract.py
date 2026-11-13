"""Настоящий HTTP/SSE без моделей: контракт обработчиков и завершение потоков."""
import json,threading,unittest,urllib.request,urllib.error,http.client
from llm_app.server import App,LimitedHTTPServer,make_handler
from .test_f12 import Tk,Eng
class HTTPTest(unittest.TestCase):
    def setUp(self):
        text='answer'
        self.app=App(Eng(len(text)),Tk(list(text)),'test',api_key='secret')
        self.app.build_ids=lambda *a:[1];self.app.tk.encode=lambda *a,**kw:[1]
        self.server=LimitedHTTPServer(('127.0.0.1',0),make_handler(self.app))
        self.worker=threading.Thread(target=lambda:self.server.serve_forever(poll_interval=.01),name='http-test')
        self.worker.start();self.url='http://127.0.0.1:'+str(self.server.server_address[1])
    def tearDown(self):
        self.server.shutdown();self.worker.join(2);self.server.server_close();self.assertFalse(self.worker.is_alive())
    def request(self,path,body,key='secret'):
        req=urllib.request.Request(self.url+path,json.dumps(body).encode(),{'Content-Type':'application/json','Authorization':'Bearer '+key})
        with urllib.request.urlopen(req,timeout=2) as r:return r.status,r.headers,r.read().decode()
    def test_all_endpoints_stream_and_nonstream(self):
        paths=['/v1/chat/completions','/v1/completions','/v1/messages','/api/chat','/api/generate']
        for path in paths:
            for stream in (False,True):
                body={'model':'test','stream':stream,'messages':[{'role':'user','content':'hi'}],'prompt':'hi','temperature':0}
                with self.subTest(path=path,stream=stream):
                    code,headers,text=self.request(path,body);self.assertEqual(code,200)
                    if not stream:self.assertIn('answer',text);json.loads(text)
                    elif path.startswith('/api/'):
                        chunks=[json.loads(line) for line in text.splitlines()]
                        self.assertTrue(chunks[-1]['done']);self.assertTrue(all('created_at' in c for c in chunks))
                    else:
                        self.assertIn('text/event-stream',headers['Content-Type']);self.assertIn('data: ',text)
                        self.assertNotIn('HTTP/1.1',text)
    def test_auth_and_bad_request(self):
        with self.assertRaises(urllib.error.HTTPError) as e:self.request('/v1/chat/completions',{},key='wrong')
        self.assertEqual(e.exception.code,401)
        with self.assertRaises(urllib.error.HTTPError) as e:self.request('/v1/chat/completions',[])
        self.assertEqual(e.exception.code,400)
    def test_failed_post_auth_does_not_poison_keepalive(self):
        client=http.client.HTTPConnection('127.0.0.1',self.server.server_address[1],timeout=2)
        try:
            client.request('POST','/v1/chat/completions',body='{"prompt":"hi"}',headers={'Authorization':'Bearer wrong','Content-Type':'application/json'})
            first=client.getresponse();self.assertEqual(first.status,401)
            self.assertEqual(first.getheader('Connection'),'close');first.read()
            client.request('GET','/v1/models',headers={'Authorization':'Bearer secret'})
            second=client.getresponse();self.assertEqual(second.status,200);second.read()
        finally:client.close()
if __name__=='__main__':unittest.main()
