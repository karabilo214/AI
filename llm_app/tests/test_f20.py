import json,unittest
import numpy as np
from llm_app.constrain import Constraint,_schema
from llm_app.server import _schema_from_openai
class SchemaTest(unittest.TestCase):
    def test_object_schema_and_validation(self):
        schema=_schema_from_openai({'type':'json_object'})
        self.assertEqual(schema,{'type':'object'})
        c=object.__new__(Constraint);c.schema=schema
        self.assertEqual(c.validate('{"x":1}'),{'x':1})
        for text in ('[]','{"x":'):
            with self.assertRaises(ValueError):c.validate(text)
    def test_schema_limits(self):
        for schema in ({'$ref':'https://example.org/schema'},{'type':'not-a-type'},{'description':'x'*40000}):
            with self.assertRaises(ValueError):_schema(schema)
    def test_rejected_token_is_error(self):
        c=object.__new__(Constraint)
        class Matcher:
            def is_terminated(self):return False
            def accept_token(self,t):return False
        c.m=Matcher()
        with self.assertRaises(RuntimeError):c.commit(0,0,1)
    def test_eos_in_draft_initializes_all_mask_rows(self):
        class Mask(np.ndarray):
            def to(self,*args,**kwargs):return self
        class Matcher:
            ended=False
            def fill_next_token_bitmask(self,bm,row):bm[row]=7
            def is_terminated(self):return self.ended
            def accept_token(self,token):self.ended=True;return True
        c=object.__new__(Constraint);c.m=Matcher();c.bm=np.full((17,4),-123,dtype=np.int32).view(Mask)
        masks,advanced=c.masks([1,2])
        self.assertEqual(advanced,1)
        np.testing.assert_array_equal(masks,np.full((3,4),7))
if __name__=='__main__':unittest.main()
