import unittest
from llm_app.gcpu import _cpp_cpu_supported
class ISATest(unittest.TestCase):
    def test_missing_isa_never_loads_avx512(self):
        for features in ({},{'X86_V4':True},{'F16C':True},{'X86_V4':False,'F16C':True}):
            self.assertFalse(_cpp_cpu_supported(features))
        self.assertTrue(_cpp_cpu_supported({'X86_V4':True,'F16C':True}))
if __name__=='__main__':unittest.main()
