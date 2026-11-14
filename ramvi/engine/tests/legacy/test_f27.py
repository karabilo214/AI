import struct,tempfile,unittest
from pathlib import Path
import numpy as np
from ramvi_engine.formats.gguf import GGUF
class ReaderTest(unittest.TestCase):
    def file(self,path,version=3,offset=0):
        name=b'x';header=b'GGUF'+struct.pack('<IQQ',version,1,0)+struct.pack('<Q',1)+name+struct.pack('<IQQIQ',2,4,2,0,offset)
        path.write_bytes(header+b'\0'*((-len(header))%32)+np.arange(8,dtype=np.float32).tobytes())
    def test_valid_rows_and_invalid_files(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'tiny.gguf';self.file(p);g=GGUF(p)
            np.testing.assert_array_equal(g.rows('x',[1]),[[4,5,6,7]])
            with self.assertRaises(ValueError):g.rows('x',[-1])
            g.close()
            for version,offset in ((1,0),(3,99999)):
                self.file(p,version,offset)
                with self.assertRaises(ValueError):GGUF(p)
            p.write_bytes(b'GGUF')
            with self.assertRaises(ValueError):GGUF(p)
if __name__=='__main__':unittest.main()
