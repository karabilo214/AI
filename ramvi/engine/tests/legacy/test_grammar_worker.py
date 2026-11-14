import subprocess,unittest
from unittest.mock import patch
import ramvi_engine.generation.constrain as constrain
class GrammarWorkerTest(unittest.TestCase):
    def test_compile_real_small_grammar_without_cuda(self):
        xgr=constrain.xgr
        info=xgr.TokenizerInfo(['{','}',':','"','a','1'],vocab_type=xgr.VocabType.RAW)
        compiled=constrain._compile_bounded(info,'{"type":"object"}')
        self.assertIsInstance(compiled,xgr.CompiledGrammar)
    def test_timeout_becomes_controlled_error(self):
        class Info:
            def serialize_json(self):return '{}'
        with patch.object(constrain.subprocess,'run',side_effect=subprocess.TimeoutExpired('worker',.1)):
            with self.assertRaises(TimeoutError):constrain._compile_bounded(Info(),None,timeout=.1)
if __name__=='__main__':unittest.main()
