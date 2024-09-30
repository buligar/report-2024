import nengo
import numpy as np
import matplotlib.pyplot as plt
model = nengo.Network()
tau_synapse = 0.5 # should be reasonably large

with model:
    def stim_func(t):
        if t<1:
            return np.sin(t)
        else:
            return 0
    stim = nengo.Node(stim_func)

    def my_function(x):
        return x
        
    def func1(x):
        b = 0
        return (1-b)*x
        
    def func2(x):
        b = 0
        return b*x 
            
    v1 = nengo.Ensemble(
        n_neurons=100,
        dimensions = 1
    )
    v2 = nengo.Ensemble(
        n_neurons=100,
        dimensions = 1
    )
    e1 = nengo.Ensemble(
        n_neurons=100,
        dimensions = 1
    )

    nengo.Connection(stim, v1)

    n_neurons = 100
    c = nengo.Connection(
        v1.neurons,
        v2,
        transform = np.zeros((1,100)),
        learning_rule_type = nengo.PES(
    learning_rate=0.0001),
        synapse=tau_synapse)
    c2 = nengo.Connection(
        v2,
        v1,
        function=func2,
        synapse=tau_synapse
    )
    def recurrent(x):
        return x
    nengo.Connection(e1, c.learning_rule, synapse=tau_synapse)
    nengo.Connection(v1, e1, function=func1, transform=-1, synapse=tau_synapse)
    nengo.Connection(v2, e1, function=func1, synapse=tau_synapse)
    # nengo.Connection(v2, v2, function=recurrent, synapse=tau_synapse)

    sin_probe = nengo.Probe(stim, synapse=0.01)
    error_probe = nengo.Probe(e1, synapse=0.01)
    pre_probe = nengo.Probe(v1, synapse=0.01)
    post_probe = nengo.Probe(v2, synapse=0.01)

with nengo.Simulator(model) as sim:
    sim.run(150)

plt.figure()
plt.plot(sim.trange(), sim.data[sin_probe], color='r', label="sin")
plt.plot(sim.trange(), sim.data[error_probe], color='b', label="e")
plt.plot(sim.trange(), sim.data[pre_probe], color='b', label="v1")
plt.plot(sim.trange(), sim.data[post_probe], color='m', label="v2")
plt.legend(loc='best')
plt.show()
