import unittest
from llm_app.server import App,_split
from .test_f12 import Tk,Eng
class ToolsTest(unittest.TestCase):
    def test_invalid_model_output_is_text(self):
        for value in ('[]','null','1','"x"','{"name":"x","arguments":[]}','bad json'):
            full='<tool_call>'+value+'</tool_call>'
            r,c,calls,opened=_split(full,False)
            self.assertEqual(c,full);self.assertEqual(calls,[])
    def test_tools_preserve_surrounding_text_and_stream(self):
        call='<tool_call>{"name":"f","arguments":{"x":1}}</tool_call>'
        full='before '+call+' between '+call+' after'
        for chunks in ([full],list(full)):
            events=list(App(Eng(len(chunks)),Tk(chunks),'test').run([1]))
            self.assertEqual(events[-1][1]['text'],'before  between  after')
            self.assertEqual(len(events[-1][1]['tool_calls']),2)
            self.assertEqual(''.join(v for k,v in events if k=='text'),'before  between  after')
if __name__=='__main__':unittest.main()
