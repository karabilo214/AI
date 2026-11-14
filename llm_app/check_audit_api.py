"""Переходное имя перенесённого инструмента."""
from llm_app._compat import alias_module
if __name__ == "__main__":
    import runpy
    runpy.run_module('ramvi.engine.tools.checks.check_audit_api', run_name="__main__")
else:
    alias_module(__name__, 'ramvi.engine.tools.checks.check_audit_api')
