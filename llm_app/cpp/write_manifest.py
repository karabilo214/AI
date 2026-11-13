"""Связь бинарника с исходником и флагами сборки; запускается build.sh."""
import hashlib,json,sys
from pathlib import Path
binary=Path(sys.argv[1])
source=Path(__file__).with_name('qmm.cpp')
manifest={'abi':2,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
          'flags':'-O3 -march=x86-64-v4 -mf16c -std=c++17'}
binary.with_suffix(binary.suffix+'.build.json').write_text(json.dumps(manifest,indent=2)+'\n')
