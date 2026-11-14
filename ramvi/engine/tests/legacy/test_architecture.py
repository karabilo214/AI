import types,unittest
from ramvi_engine.formats.gguf import validate_qwen,eos_ids
class ArchitectureTest(unittest.TestCase):
    def valid(self):
        meta={'general.architecture':'qwen3'}
        for key,value in {'block_count':1,'embedding_length':512,'attention.head_count':4,'attention.head_count_kv':2,
                          'attention.key_length':128,'feed_forward_length':768,'rope.freq_base':1000000,
                          'attention.layer_norm_rms_epsilon':1e-6}.items():meta['qwen3.'+key]=value
        shapes={'token_embd.weight':(4,512),'output.weight':(4,512),'output_norm.weight':(1,512)}
        for name,n,k in [('attn_norm',1,512),('ffn_norm',1,512),('attn_q_norm',1,128),('attn_k_norm',1,128),
                         ('attn_q',512,512),('attn_k',256,512),('attn_v',256,512),('attn_output',512,512),
                         ('ffn_gate',768,512),('ffn_up',768,512),('ffn_down',512,768)]:shapes['blk.0.'+name+'.weight']=(n,k)
        g=types.SimpleNamespace(meta=meta,tensors=shapes,raw=lambda name:(None,*shapes[name],'F32'))
        return g,shapes
    def test_valid_and_inconsistent_shapes(self):
        g,shapes=self.valid();validate_qwen(g)
        for key in ('blk.0.attn_q.weight','output.weight','token_embd.weight'):
            original=shapes[key];shapes[key]=(original[0]+1,original[1]+1)
            with self.assertRaises(ValueError):validate_qwen(g)
            shapes[key]=original
    def test_invalid_numeric_metadata_and_eos(self):
        g,_=self.valid();g.meta['qwen3.rope.freq_base']=float('nan')
        with self.assertRaises(ValueError):validate_qwen(g)
        for value in (True,-1,'1'):
            with self.assertRaises(ValueError):eos_ids(types.SimpleNamespace(meta={'tokenizer.ggml.eos_token_id':value}))
if __name__=='__main__':unittest.main()
