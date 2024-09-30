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
    
    def my_function(x):
        return x
        
    def func1(x):
        b = 0
        return (1-b)*x
    def func2(x):
        b = 0
        return b*x 
        

    
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
    e1p = nengo.Ensemble(
        n_neurons=100,
        dimensions = 1
    )
    e2p = nengo.Ensemble(
        n_neurons=100,
        dimensions = 1
    )
    p1p = nengo.Ensemble(
        n_neurons=100,
        dimensions = 1
    )

    p2p = nengo.Ensemble(
        n_neurons=100,
        dimensions = 1
    )
    nengo.Connection(stim, e1)
    nengo.Connection(stim2, p2)
    c1 = nengo.Connection(
        e2.neurons, 
        v2,
        transform = np.zeros((1,100)),
        learning_rule_type = nengo.PES(
    learning_rate=0.0001))
    c2 = nengo.Connection(
        e1.neurons, 
        v1,
        transform = np.zeros((1,100)),
        learning_rule_type = nengo.PES(
    learning_rate=0.0001))
    c3 = nengo.Connection(
        p2.neurons, 
        v2,
        transform = np.zeros((1,100)),
        learning_rule_type = nengo.PES(
    learning_rate=0.0001))
    c4 = nengo.Connection(
        p1.neurons, 
        v1,
        transform = np.zeros((1,100)),
        learning_rule_type = nengo.PES(
    learning_rate=0.0001))
    nengo.Connection(e2p, c1.learning_rule)
    nengo.Connection(e1p, c2.learning_rule)
    nengo.Connection(p2p, c3.learning_rule)
    nengo.Connection(p1p, c4.learning_rule)

    
    nengo.Connection(e1, e1p,function=func1, transform=-1)
    nengo.Connection(e2, e2p,function=func1, transform=-1)
    nengo.Connection(v1, e1p)
    nengo.Connection(v2, e2p)
    nengo.Connection(v1, p1p)
    nengo.Connection(v2, p2p)
    nengo.Connection(v1, e2, function=func1)
    nengo.Connection(v2, p1, function=func2)
    nengo.Connection(p1, p1p,function=func2, transform=-1)
    nengo.Connection(p2, p2p,function=func2, transform=-1)



    sin_probe = nengo.Probe(stim, synapse=0.01)
    cos_probe = nengo.Probe(stim2, synapse=0.01)
    pre_probe = nengo.Probe(v1, synapse=0.01)
    post_probe = nengo.Probe(v2, synapse=0.01)

with nengo.Simulator(model) as sim:
    sim.run(500)

plt.figure()
plt.plot(sim.trange(), sim.data[sin_probe], color='r', label="sin")
plt.plot(sim.trange(), sim.data[cos_probe], color='g', label="cos")
plt.plot(sim.trange(), sim.data[pre_probe], color='b', label="v1")
plt.plot(sim.trange(), sim.data[post_probe], color='m', label="v2")
plt.legend(loc='best')
plt.show()
