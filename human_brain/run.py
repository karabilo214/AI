from Brain import Brain

# =============================================
# Создаём маленькую нейронную сеть
# =============================================

brain = Brain()

brain.add_neuron("A")
brain.add_neuron("B")
brain.add_neuron("C")

ab = brain.connect("A", "B")
bc = brain.connect("B", "C")

# Стимулируем только нейрон A
brain.stimulate("A", 20.0)

# Мозг работает самостоятельно следующие 12 шагов
brain.run(12)

print("Вес синапса A->B:", ab.synapse.weight)
print("Миелин A->B:", ab.myelin.thickness)
print("Импульсов нейрона C:", brain.neurons["C"].spikes)
