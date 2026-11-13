import unittest
from llm_app.request_validation import validate_capabilities
class CapabilitiesTest(unittest.TestCase):
    def test_rejects_unimplemented_semantics(self):
        for req in ({'model':'other'},{'n':2},{'options':{'top_k':5}}, {'presence_penalty':1},
                    {'messages':[{'content':[{'type':'image_url'}]}]}):
            with self.assertRaises((ValueError,NotImplementedError)):validate_capabilities(req,'test',2048,'/v1/chat/completions')
        validate_capabilities({'model':'test','options':{'temperature':0,'top_k':1}},'test',2048,'/api/chat')
if __name__=='__main__':unittest.main()
