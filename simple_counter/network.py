"""Настоящая маленькая нейросеть: сама учится складывать и вычитать в пределах 100.

    python3 network.py train         учится и показывает результат
    python3 network.py 37 + 45       отвечает на пример

Сеть одна на оба действия. Она получает три числа: первое число, второе и знак действия.
Что такое сложение, ей никто не говорит. В ответе сети нет ни одной строки с a + b: эта операция
есть только у учителя, в функции make_examples. Предел 100 сеть тоже не проверяет.
"""
import json
import math
import random
import sys
from pathlib import Path

BRAIN_FILE = Path(__file__).parent / "network_brain.json"
HIDDEN = 15           # сколько нейронов в скрытом слое
ROUNDS = 4000         # сколько раз просматривает все учебные примеры
RATE_START = 0.05     # сила поправок в начале
RATE_END = 0.002      # сила поправок в конце
TRAIN_SHARE = 0.2     # доля примеров для обучения, остальные сеть не видит


# УЧИТЕЛЬ. Это единственное место, где знают правильный ответ.

def make_examples():
    examples = []
    for a in range(101):
        for b in range(101):
            if a + b <= 100:
                examples.append((a, b, 1.0, a + b))
            if a - b >= 0:
                examples.append((a, b, -1.0, a - b))
    return examples


# СЕТЬ. Каждый скрытый нейрон складывает входы с весами и пропускает через tanh:
# плавный изгиб, без которого сеть была бы одной прямой линией. Выходной нейрон
# складывает сигналы скрытых нейронов.

def new_network():
    hidden = [[random.uniform(-1, 1) for _ in range(4)] for _ in range(HIDDEN)]  # 3 входа + смещение
    output = [random.uniform(-1, 1) for _ in range(HIDDEN + 1)]                  # все скрытые + смещение
    return {"hidden": hidden, "output": output}


def think(network, inputs):
    signals = [
        math.tanh(sum(w * x for w, x in zip(neuron, inputs)) + neuron[3])
        for neuron in network["hidden"]
    ]
    result = sum(w * s for w, s in zip(network["output"], signals)) + network["output"][-1]
    return signals, result


# ОБУЧЕНИЕ. Обратное распространение ошибки: сначала правим выходной нейрон, потом
# каждому скрытому передаём его долю вины за ошибку.

def learn(network, inputs, target, rate):
    signals, result = think(network, inputs)
    error = result - target
    output = network["output"]
    for i, neuron in enumerate(network["hidden"]):
        blame = error * output[i] * (1 - signals[i] ** 2)  # доля вины этого нейрона
        for j in range(3):
            neuron[j] -= rate * blame * inputs[j]
        neuron[3] -= rate * blame
        output[i] -= rate * error * signals[i]
    output[-1] -= rate * error


def to_inputs(a, b, sign):
    return (a / 100, b / 100, sign)


def answer(network, a, b, sign):
    return round(think(network, to_inputs(a, b, sign))[1] * 100)


def check(network, examples):
    right = near = 0
    for a, b, sign, target in examples:
        got = answer(network, a, b, sign)
        right += got == target
        near += abs(got - target) <= 1
    return right / len(examples), near / len(examples)


def train(seed=0):
    random.seed(seed)
    examples = make_examples()
    random.shuffle(examples)
    cut = int(len(examples) * TRAIN_SHARE)
    studied, unseen = examples[:cut], examples[cut:]
    network = new_network()
    print(f"Учится на {len(studied)} примерах, проверка на {len(unseen)} других")
    print(f"до обучения: точно {check(network, unseen)[0]:.1%}")
    for round_number in range(1, ROUNDS + 1):
        rate = RATE_START * (RATE_END / RATE_START) ** (round_number / ROUNDS)
        random.shuffle(studied)
        for a, b, sign, target in studied:
            learn(network, to_inputs(a, b, sign), target / 100, rate)
        if round_number % 100 == 0:
            exact, near = check(network, unseen)
            print(f"круг {round_number}: точно {exact:.1%}, с ошибкой не больше 1: {near:.1%}", flush=True)
    BRAIN_FILE.write_text(json.dumps(network))
    return network, unseen


def ask(a, operation, b):
    if not BRAIN_FILE.exists():
        print("Сеть ещё не обучена. Сначала: python3 network.py train")
        return
    network = json.loads(BRAIN_FILE.read_text())
    print(f"{a} {operation} {b} = {answer(network, a, b, 1.0 if operation == '+' else -1.0)}")


if __name__ == "__main__":
    if sys.argv[1:] == ["train"]:
        train()
    elif len(sys.argv) == 4:
        ask(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
    else:
        print(__doc__)
