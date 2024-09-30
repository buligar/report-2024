import nest
import numpy as np
import matplotlib.pyplot as plt

# Функция для запуска симуляции с заданными параметрами и возвращения последнего времени спайка
def run_simulation(n_neurons, w1):
    nest.ResetKernel()
    
    nest.SetKernelStatus({
      "local_num_threads": 1,
      "resolution": 0.1,
      "rng_seed": 1
    })
    
    # Создание нейронов и устройств
    n1 = nest.Create("iaf_psc_alpha", n_neurons, params={"tau_m": 10})
    n2 = nest.Create("iaf_psc_alpha", n_neurons, params={"tau_m": 10})
    sg1 = nest.Create("spike_generator", 1, params={"spike_times": [1], "spike_weights": [10000]})
    ng1 = nest.Create("noise_generator", 1, params={"mean": 0, "std": 300})
    vm1 = nest.Create("voltmeter", 1)
    sr1 = nest.Create("spike_recorder", 1)
    vm2 = nest.Create("voltmeter", 1)
    sr2 = nest.Create("spike_recorder", 1)
    
    # Подключение нейронов и устройств
    nest.Connect(n1, n1, conn_spec={"rule": "fixed_indegree", "indegree": 10})
    nest.Connect(n2, n2, conn_spec={"rule": "fixed_indegree", "indegree": 10}, syn_spec={"weight": 1})
    nest.Connect(n1, n2, conn_spec={"rule": "fixed_indegree", "indegree": 5}, syn_spec={"weight": w1})
    nest.Connect(n2, n1, conn_spec={"rule": "fixed_indegree", "indegree": 5}, syn_spec={"weight": w1})
    nest.Connect(sg1, n1)
    nest.Connect(ng1, n1)
    nest.Connect(ng1, n2)
    nest.Connect(vm1, n1)
    nest.Connect(vm2, n2)
    nest.Connect(n1, sr1)
    nest.Connect(n2, sr2)
    
    # Запуск симуляции
    nest.Simulate(1000)
    
    # Получение последнего времени спайка
    sr1_events = np.array(sr1.events['times'])
    last_spike_time = sr1_events[-1] if len(sr1_events) > 0 else 0
    return last_spike_time

# Диапазоны параметров
n_neurons_range = range(1, 501, 10)
w1_range = range(230, 301, 1)

# Массивы для хранения результатов
results = []

# Циклы по диапазонам параметров
for n_neurons in n_neurons_range:
    for w1 in w1_range:
        last_spike_time = run_simulation(n_neurons, w1)
        model_type = "Нестабильное фазовое состояние" if last_spike_time > 995 else "Стабильное фазовое состояние"
        results.append((n_neurons, w1, model_type))

# Построение графика
fig, ax = plt.subplots()
oscillatory = [(n, w) for n, w, model in results if model == "Нестабильное фазовое состояние"]
non_oscillatory = [(n, w) for n, w, model in results if model == "Стабильное фазовое состояние"]

ax.scatter(*zip(*oscillatory), c='r', label="Нестабильное фазовое состояние")
ax.scatter(*zip(*non_oscillatory), c='b', label="Стабильное фазовое состояние")

ax.set_xlabel('Number of Neurons')
ax.set_ylabel('Weight')
ax.legend()
plt.show()
