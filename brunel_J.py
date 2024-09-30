import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from scipy.optimize import curve_fit

# Настройка шрифтов для графиков
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18, 'legend.fontsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12})

def sim(g, nu_ext_over_nu_thr, sim_time, epsilon, J, D, plot=False):
    start_scope()
    # network parameters
    N_E = 1000
    gamma = 0.25
    N_I = round(gamma * N_E)
    N = N_E + N_I
    C_E = epsilon * N_E
    C_ext = C_E

    # neuron parameters
    tau = 20 * ms
    theta = 20 * mV
    V_r = 10 * mV
    tau_rp = 2 * ms

    # synapse parameters
    D = 1.5 * ms

    # external stimulus
    nu_thr = theta / (J * C_E * tau)
    nu_ext = nu_ext_over_nu_thr * nu_thr

    defaultclock.dt = 0.1 * ms

    neurons = NeuronGroup(N,
                          "dv/dt = -v/tau : volt (unless refractory)",
                          threshold="v > theta",
                          reset="v = V_r",
                          refractory=tau_rp,
                          method="exact")
    excitatory_neurons = neurons[:N_E]
    inhibitory_neurons = neurons[N_E:]

    exc_synapses = Synapses(excitatory_neurons, target=neurons, on_pre="v += J", delay=D)
    exc_synapses.connect(p=epsilon)

    inhib_synapses = Synapses(inhibitory_neurons, target=neurons, on_pre="v += -g*J", delay=D)
    inhib_synapses.connect(p=epsilon)

    external_poisson_input = PoissonInput(
        target=neurons, target_var="v", N=C_ext, rate=nu_ext, weight=J
    )

    rate_monitor = PopulationRateMonitor(neurons)
    spike_monitor = SpikeMonitor(neurons[:50])

    # Запуск симуляции
    run(sim_time)

    rate = rate_monitor.rate/Hz - np.mean(rate_monitor.rate/Hz)

    # Функция Гаусса для подгонки
    def func(x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    N = len(rate_monitor.t) # число выборок
    sampling_rate = 10000 # 1/defaultclock.dt   
    max_point = int(N*300/sampling_rate)
    x = rfftfreq(N, d=1/sampling_rate)
    x = x[:max_point]
    yn = 2*np.abs(rfft(rate))/N
    yn = yn[:max_point]

    max_rate = np.argmax(yn)

    if plot:
        plt.figure(figsize=(12, 6))
        plt.suptitle(f'D = {g}')

        # Первый график (спайки)
        plt.subplot(311)
        plt.plot(spike_monitor.t / ms, spike_monitor.i, '|')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron index')
        plt.xlim([0, 250])

        # Второй график (частота)
        plt.subplot(312)
        plt.plot(rate_monitor.t / ms, rate_monitor.rate / Hz)
        plt.xlabel('Time (ms)')
        plt.ylabel('Rate (Hz)')
        plt.xlim([0, 250])

        # Третий график (функция)
        plt.subplot(313)
        plt.plot(x, yn, c='k', label='Function')
        plt.xlabel('Rate (Hz)')
        plt.xlim([0, 250])
        plt.show()

    return rate_monitor, spike_monitor, max_rate

# Параметры симуляции
tests = np.arange(0, 10, 1)
J_values = np.arange(0.05, 1.05, 0.1)

g = 1
sim_time = 1000 * ms
epsilon = 0.1
nu_ext_over_nu_thr = 2
D = 1.5 * ms

mean_rates = []
std_rates = []

for J in J_values:
    print('J', round(J, 2))
    max_rates_for_g = []
    for n_test in tests:
        print('n_test', n_test)
        rate_monitor, spike_monitor, max_rate = sim(g, nu_ext_over_nu_thr, sim_time, epsilon, J * mV, D)
        max_rates_for_g.append(max_rate)
    
    mean_rates.append(np.mean(max_rates_for_g))
    std_rates.append(np.std(max_rates_for_g))

# Построение графика
plt.figure(figsize=(12, 8))
plt.errorbar(J_values, mean_rates, yerr=std_rates, fmt='o', capsize=5, label='Average max rate with std deviation')
plt.ylabel('Rate (Hz)')
plt.xlabel('J (mV)')
plt.title(f'g={g}, epsilon={epsilon}, nu_ext_over_nu_thr={nu_ext_over_nu_thr}, D={D}')
plt.tight_layout()
plt.legend()
plt.show()

