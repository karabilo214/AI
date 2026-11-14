import types,unittest
from ramvi_engine.server.app import App
class Tk:
    def __init__(self,chunks):self.chunks=chunks
    def decode(self,tokens,**kw):return ''.join(self.chunks[t] for t in tokens)
class Eng:
    ck=types.SimpleNamespace(eos={-1})
    def __init__(self,n):self.n=n
    def generate(self,*a,**kw):yield from range(self.n)
class StreamTest(unittest.TestCase):
    def test_all_two_piece_splits_and_character_tokens(self):
        cases=[('<think>abc</think>Ответ',False,'abc','Ответ'),('abc</think>Ответ',True,'abc','Ответ'),
               ('a'*40+'</think>Ответ',False,'','a'*40+'Ответ'),('Обычный ответ',False,'','Обычный ответ')]
        for text,think,r,c in cases:
            variants=[[text[:i],text[i:]] for i in range(1,len(text))]+[list(text)]
            for chunks in variants:
                with self.subTest(text=text,chunks=chunks):
                    events=list(App(Eng(len(chunks)),Tk(chunks),'test').run([1],think=think))
                    fin=events[-1][1]
                    self.assertEqual(''.join(v for k,v in events if k=='text'),c)
                    self.assertEqual(''.join(v for k,v in events if k=='reasoning'),r)
                    self.assertEqual((fin['reasoning'],fin['text']),(r,c))
if __name__=='__main__':unittest.main()
