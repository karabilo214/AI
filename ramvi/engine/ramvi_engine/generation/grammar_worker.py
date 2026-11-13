"""Одноразовый CPU-процесс компиляции грамматики; родитель ограничивает время жизни."""
import json,sys
from pathlib import Path
import xgrammar as xgr

def main():
    request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    info=xgr.TokenizerInfo.deserialize_json(request['tokenizer'])
    compiler=xgr.GrammarCompiler(info,max_threads=1)
    schema=request['schema']
    grammar=compiler.compile_builtin_json_grammar() if schema is None else compiler.compile_json_schema(schema)
    Path(sys.argv[2]).write_text(grammar.serialize_json(), encoding='utf-8')
if __name__=='__main__':main()
