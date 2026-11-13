import json,unittest
from llm_app.server import openai_messages,anthropic_messages
class ToolIdentityTest(unittest.TestCase):
    def test_reordered_results_same_name_keep_identity(self):
        calls=[{'id':i,'function':{'name':'f','arguments':'{}'}} for i in ('a','b')]
        msgs=[{'role':'assistant','content':'','tool_calls':calls}]+[{'role':'tool','tool_call_id':i,'content':i} for i in ('b','a')]
        out=openai_messages(msgs)
        self.assertEqual([json.loads(m['content'])['tool_call_id'] for m in out[1:]],['b','a'])
        self.assertEqual(out[0]['tool_calls'][0]['id'],'a')
    def test_anthropic_error_and_identity(self):
        req={'messages':[{'role':'assistant','content':[{'type':'tool_use','id':'a','name':'f','input':{}}]},
                         {'role':'user','content':[{'type':'tool_result','tool_use_id':'a','is_error':True,'content':'failed'}]}]}
        obj=json.loads(anthropic_messages(req)[1]['content'])
        self.assertEqual(obj['tool_call_id'],'a');self.assertTrue(obj['is_error'])
if __name__=='__main__':unittest.main()
