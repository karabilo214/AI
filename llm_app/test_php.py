"""Проверка PHP-ответов запуском: python -m llm_app.test_php ask5_<имя>.json (нужен php в PATH)."""
import json
import re
import subprocess
import sys
import tempfile

TESTS = [
    # 1) римские числа
    r'''
function ref_roman($n) { $m = [1000=>'M',900=>'CM',500=>'D',400=>'CD',100=>'C',90=>'XC',50=>'L',40=>'XL',10=>'X',9=>'IX',5=>'V',4=>'IV',1=>'I'];
  $s = ''; foreach ($m as $v => $r) { while ($n >= $v) { $s .= $r; $n -= $v; } } return $s; }
$bad = 0; $first = '';
for ($n = 1; $n <= 3999; $n++) {
  $r = int_to_roman($n); $ok = ($r === ref_roman($n)) && (roman_to_int(ref_roman($n)) === $n);
  if (!$ok) { $bad++; if ($first === '') $first = "$n -> " . var_export($r, true) . " / " . var_export(roman_to_int(ref_roman($n)), true); }
}
echo $bad === 0 ? "OK: 3999/3999\n" : "FAIL: $bad ошибок, первая: $first\n";
''',
    # 2) CSV
    r'''
$cases = ['a,b,c', '"a,b",c', '"he said ""hi""",x', ',,', '"",""', 'a,"",b', '"multi, comma","quote "" inside",end',
          ' spaced , x ', '"a"', 'x,', ',x', '"""",a', '"a""b""c"', 'one', '"x,y,z"', '1,"2,3",4,"5""6"'];
$bad = 0; $first = '';
foreach ($cases as $c) {
  $want = str_getcsv($c, ',', '"', ''); $got = parse_csv_line($c);
  if ($got !== $want) { $bad++; if ($first === '') $first = $c . ' => ' . json_encode($got) . ' (нужно ' . json_encode($want) . ')'; }
}
echo $bad === 0 ? "OK: " . count($cases) . "/" . count($cases) . "\n" : "FAIL: $bad из " . count($cases) . ", первая: $first\n";
''',
    # 3) LRU
    r'''
mt_srand(1); $bad = 0; $first = ''; $ops = 0;
foreach ([1, 2, 3, 10, 50] as $cap) {
  $c = new LRUCache($cap); $ref = [];  // эталон: массив в порядке использования (конец — самый свежий)
  for ($i = 0; $i < 4000; $i++) {
    $k = mt_rand(0, 2 * $cap + 3); $ops++;
    if (mt_rand(0, 1)) { $v = mt_rand(0, 1000);
      $c->put($k, $v); unset($ref[$k]); $ref[$k] = $v; if (count($ref) > $cap) { reset($ref); unset($ref[key($ref)]); }
    } else {
      $want = array_key_exists($k, $ref) ? $ref[$k] : -1;
      if ($want !== -1) { unset($ref[$k]); $ref[$k] = $want; }
      $got = $c->get($k);
      if ($got !== $want) { $bad++; if ($first === '') $first = "ёмкость $cap, шаг $i: get($k) = " . var_export($got, true) . ", нужно $want"; }
    }
  }
}
echo $bad === 0 ? "OK: $ops операций\n" : "FAIL: $bad ошибок из $ops, первая: $first\n";
''',
]


def extract(out):
    m = re.findall(r"```(?:php)?\s*\n(.*?)```", out, re.S)
    code = "\n".join(m) if m else out
    code = re.sub(r"^\s*<\?php", "", code.strip())
    return code.replace("?>", "")


def main():
    r = json.load(open(sys.argv[1], encoding="utf-8"))
    for i, (e, t) in enumerate(zip(r, TESTS)):
        src = "<?php\n" + extract(e["out"]) + "\n" + t
        with tempfile.NamedTemporaryFile("w", suffix=".php", delete=False, encoding="utf-8") as f:
            f.write(src)
        p = subprocess.run(["php", f.name], capture_output=True, text=True, timeout=120)
        res = (p.stdout.strip().splitlines() or [""])[-1] if p.returncode == 0 else "ОШИБКА PHP: " + (p.stderr or p.stdout).strip()[:300]
        print(f"задача {i + 1}: {res}")


if __name__ == "__main__":
    main()
