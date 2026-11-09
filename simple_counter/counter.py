"""Самый простой ИИ: учится складывать и вычитать в пределах 100.

Запуск:
    python3 counter.py train         учится и показывает результат
    python3 counter.py 37 + 45       отвечает на пример

Правила ему никто не говорит. Он видит только примеры с ответами и подбирает три числа.
"""
import json
import random
import sys
from pathlib import Path

BRAIN_FILE = Path(__file__).parent / "brain.json"
TRAIN_SHARE = 0.2     # на скольких примерах учится (остальные он никогда не видел)
ROUNDS = 300          # сколько раз просматривает все учебные примеры
LEARNING_RATE = 0.5   # насколько сильно исправляется после каждой ошибки


# 1. ВСЕ ПРИМЕРЫ. Каждый пример: первое число, второе число, правильный ответ.

def all_examples(operation):
    examples = []
    for a in range(101):
        for b in range(101):
            answer = a + b if operation == "+" else a - b
            if 0 <= answer <= 100:
                examples.append((a, b, answer))
    return examples


# 2. НЕЙРОН. Вся его "память" это три числа: w1, w2, w3.
#    Числа делим на 100, чтобы работать с маленькими значениями: так обучение не разносит.

def guess(neuron, a, b):
    w1, w2, w3 = neuron
    return w1 * (a / 100) + w2 * (b / 100) + w3


# 3. ОБУЧЕНИЕ. Угадал, посмотрел на ошибку, чуть поправил каждое число
#    в ту сторону, где ошибка станет меньше. Чем больше вход, тем больше он виноват.

def learn_one_example(neuron, a, b, answer):
    error = guess(neuron, a, b) - answer / 100
    neuron[0] -= LEARNING_RATE * error * (a / 100)
    neuron[1] -= LEARNING_RATE * error * (b / 100)
    neuron[2] -= LEARNING_RATE * error


def train_neuron(examples):
    neuron = [random.uniform(-1, 1) for _ in range(3)]  # сначала случайные числа: он ничего не знает
    for _ in range(ROUNDS):
        random.shuffle(examples)
        for a, b, answer in examples:
            learn_one_example(neuron, a, b, answer)
    return neuron


# 4. ОТВЕТ. Переводим обратно в обычное число и округляем.

def answer_for(neuron, a, b):
    return round(guess(neuron, a, b) * 100)


# 5. ПРОВЕРКА только на примерах, которых не было в обучении.

def share_correct(neuron, examples):
    right = sum(answer_for(neuron, a, b) == answer for a, b, answer in examples)
    return right / len(examples)


def train():
    random.seed(0)
    brain = {}
    for operation in "+-":
        examples = all_examples(operation)
        random.shuffle(examples)
        cut = int(len(examples) * TRAIN_SHARE)
        for_training, for_testing = examples[:cut], examples[cut:]

        untrained = [random.uniform(-1, 1) for _ in range(3)]
        before = share_correct(untrained, for_testing)
        neuron = train_neuron(for_training)
        after = share_correct(neuron, for_testing)
        brain[operation] = neuron

        print(f"Операция {operation}: учился на {len(for_training)} примерах, проверка на {len(for_testing)} других")
        print(f"  верно до обучения:    {before:.1%}")
        print(f"  верно после обучения: {after:.1%}")
        print(f"  его память: w1={neuron[0]:.3f}  w2={neuron[1]:.3f}  w3={neuron[2]:.3f}")
    BRAIN_FILE.write_text(json.dumps(brain, indent=1))
    print(f"Память сохранена в {BRAIN_FILE.name}")


def ask(a, operation, b):
    if not BRAIN_FILE.exists():
        print("Я ещё не обучен. Сначала: python3 counter.py train")
        return
    if operation not in ("+", "-"):
        print("Я умею только + и -")
        return
    result = a + b if operation == "+" else a - b
    if not (0 <= a <= 100 and 0 <= b <= 100 and 0 <= result <= 100):
        print("Это вне пределов 100, такому я не учился")
        return
    brain = json.loads(BRAIN_FILE.read_text())
    print(f"{a} {operation} {b} = {answer_for(brain[operation], a, b)}")


if __name__ == "__main__":
    if sys.argv[1:] == ["train"]:
        train()
    elif len(sys.argv) == 4:
        ask(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
    else:
        print(__doc__)
