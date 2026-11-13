import types,unittest
from llm_app.tokenizer_check import validate_tokenizer,validate_draft
class Tk:
    special_tokens_map={'eos_token':'end'}
    def get_vocab(self):return {'a':0,'end':1}
class TokenizerTest(unittest.TestCase):
    def test_mismatch_rejected(self):
        e=types.SimpleNamespace(g=types.SimpleNamespace(meta={'tokenizer.ggml.tokens':['a','end']}),V=2,ck=types.SimpleNamespace(eos={1}))
        self.assertEqual(len(validate_tokenizer(e,Tk())),64)
        e.g.meta['tokenizer.ggml.tokens']=['end','a']
        with self.assertRaises(ValueError):validate_tokenizer(e,Tk())
if __name__=='__main__':unittest.main()
