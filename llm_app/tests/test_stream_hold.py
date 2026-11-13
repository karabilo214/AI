import types, unittest
from llm_app.server import App, _held


class Tk:
    def __init__(self, chunks): self.chunks = chunks
    def decode(self, tokens, **kw): return ''.join(self.chunks[t] for t in tokens)


class Eng:
    """Отдаёт слова по одному и запоминает, сколько уже отдано, — чтобы видеть, когда клиент получил текст."""
    ck = types.SimpleNamespace(eos={-1})
    def __init__(self, n): self.n, self.given = n, 0
    def generate(self, *a, **kw):
        for t in range(self.n):
            self.given = t + 1
            yield t


class HoldTest(unittest.TestCase):
    def test_held_only_marker_prefix(self):
        self.assertEqual(_held("Hello", ["<tool_call>"]), 0)
        self.assertEqual(_held("Hello <tool", ["<tool_call>"]), 5)
        self.assertEqual(_held("a<", ["<tool_call>", "</think>"]), 1)
        self.assertEqual(_held("x END", ["ENDING"]), 3)
        self.assertEqual(_held("", ["<tool_call>"]), 0)

    def test_first_word_sent_immediately(self):
        chunks = ["Hello", " world", "!"]
        eng = Eng(len(chunks))
        when = []
        for kind, value in App(eng, Tk(chunks), "test")._run([1]):
            if kind == "text":
                when.append((eng.given, value))
        self.assertEqual(when[0], (1, "Hello"))  # первое слово — сразу после первого шага движка

    def test_tool_call_and_stop_still_hidden(self):
        chunks = ["Ок", " <", "tool", "_call>", '{"name":"f","arguments":{}}', "</tool_call>"]
        events = list(App(Eng(len(chunks)), Tk(chunks), "test").run([1]))
        self.assertEqual("".join(v for k, v in events if k == "text"), "Ок ")
        self.assertEqual(events[-1][1]["tool_calls"], [{"name": "f", "arguments": {}}])
        chunks = ["abc", " ST", "OP", " tail"]
        events = list(App(Eng(len(chunks)), Tk(chunks), "test").run([1], stop=["STOP"]))
        self.assertEqual("".join(v for k, v in events if k == "text"), "abc ")


if __name__ == "__main__":
    unittest.main()
