import types,unittest
from ramvi_engine.server.app import App,make_handler
class CompletionTest(unittest.TestCase):
    def test_stream_and_prompt_shapes(self):
        app=App(None,types.SimpleNamespace(encode=lambda *a,**k:[1]),'test')
        fin=dict(text='answer',finish='stop',prompt_n=1,gen_n=1)
        seen=[]
        def run(ids,*a,**k):seen.append(ids);yield 'text','answer';yield 'end',fin
        app.run=run;h=object.__new__(make_handler(app));events=[]
        h._start_stream=lambda *a:None;h._sse=lambda obj:events.append(obj);h._json=lambda obj:obj
        h.openai_completion({'prompt':[3,4],'stream':True})
        self.assertEqual(seen,[[3,4]]);self.assertEqual(events[-1],'[DONE]');self.assertEqual(events[0]['choices'][0]['text'],'answer')
        for prompt in ([],['a','b'],[[]]):
            with self.assertRaises(ValueError):h.openai_completion({'prompt':prompt})
if __name__=='__main__':unittest.main()
