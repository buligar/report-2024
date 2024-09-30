import nengo
import numpy as np
import matplotlib.pyplot as plt
model = nengo.Network()

with model:
    def stim_func(t):
        return np.sin(t*2*np.pi)
    stim = nengo.Node(stim_func)
    def stim_func2(t):
        return np.cos(t*2*np.pi)
    stim2 = nengo.Node(stim_func2)
    
    def my_function(x):
        return x
        
    def func1(x):
        b = 1
        return (1-b)*x
        
    def func2(x):
        b = 1
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
    e2 = nengo.Ensemble(
        n_neurons=100,
        dimensions = 1
    )
    p1 = nengo.Ensemble(
        n_neurons=100,
        dimensions = 1
    )
    p2 = nengo.Ensemble(
        n_neurons=100,
        dimensions = 1
    )
    
    nengo.Connection(stim, p1)
    nengo.Connection(stim2, p2)
    
    c = nengo.Connection(
        v1,
        v2,
        function=func1,
        learning_rule_type = nengo.PES(
    learning_rate=0.0001))
    c2 = nengo.Connection(
        v2,
        v1,
        function=func2,
        learning_rule_type = nengo.PES(
    learning_rate=0.0001))
    
    nengo.Connection(e1, c.learning_rule)
    nengo.Connection(p1, v1, function=func1)
    nengo.Connection(v1, e1, function=func1, transform=-1)
    nengo.Connection(v2, e1, function=func1)
    nengo.Connection(p2, v2, function=func2)
    nengo.Connection(e2, c.learning_rule)
    nengo.Connection(v1, e2, function=func2)
    nengo.Connection(v2, e2, function=func2, transform=-1)

    sin_probe = nengo.Probe(stim, synapse=0.01)
    cos_probe = nengo.Probe(stim2, synapse=0.01)
    pre_probe = nengo.Probe(v1, synapse=0.01)
    post_probe = nengo.Probe(v2, synapse=0.01)

with nengo.Simulator(model) as sim:
    sim.run(15)

plt.figure()
plt.plot(sim.trange(), sim.data[sin_probe], color='r', label="sin")
plt.plot(sim.trange(), sim.data[cos_probe], color='g', label="cos")
plt.plot(sim.trange(), sim.data[pre_probe], color='b', label="v1")
plt.plot(sim.trange(), sim.data[post_probe], color='m', label="v2")
plt.legend(loc='best')
plt.show()
