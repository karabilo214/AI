import unittest
from llm_app.server import App
from .test_f12 import Tk,Eng
class StopTest(unittest.TestCase):
    def test_earliest_stop_independent_of_order_and_token_split(self):
        text='before EARLY middle LATE after'
        for stops in (['LATE','EARLY'],['EARLY','LATE']):
            for chunks in ([text],list(text),['before EA','RLY middle LATE after']):
                events=list(App(Eng(len(chunks)),Tk(chunks),'test').run([1],stop=stops))
                self.assertEqual(events[-1][1]['text'],'before ')
                self.assertEqual(''.join(v for k,v in events if k=='text'),'before ')
if __name__=='__main__':unittest.main()
