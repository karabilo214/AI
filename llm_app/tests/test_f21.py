import io,types,unittest,json
from llm_app.server import App,make_handler
from .test_f12 import Tk,Eng
class OllamaTest(unittest.TestCase):
    def test_stream_thinking_not_duplicated_and_has_timestamp(self):
        app=App(None,None,'test');app.build_ids=lambda *a:[1];app.constraint=lambda *a:None
        fin=dict(text='answer',reasoning='thought',tool_calls=[],finish='stop',total_s=1,prompt_n=1,
                 prompt_ms=1,gen_n=3,decode_n=2,gen_ms=1)
        app.run=lambda *a:iter([('reasoning','thought'),('text','answer'),('end',fin)])
        h=object.__new__(make_handler(app));h._start_stream=lambda *a:None;h.wfile=io.BytesIO()
        h.ollama({'stream':True},True)
        chunks=[json.loads(line) for line in h.wfile.getvalue().splitlines()]
        self.assertEqual(''.join(c['message'].get('thinking','') for c in chunks),'thought')
        self.assertTrue(all('created_at' in c for c in chunks))
    def test_eos_is_counted_in_usage(self):
        eng=Eng(1);eng.ck=types.SimpleNamespace(eos={0})
        events=list(App(eng,Tk(['']), 'test').run([1]))
        self.assertEqual(events[-1][1]['gen_n'],1)
if __name__=='__main__':unittest.main()
