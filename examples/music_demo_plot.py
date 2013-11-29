


from glob import glob
from neo.io import get_io
from pyNN.utility.plotting import Figure, Panel

data = {}
for filename in glob("Results/music_demo_*.pkl"):
    block = get_io(filename).read_block()
    data[block.name] = block.segments[0]

panels = [
    Panel(data["input (NEST)"].spiketrains),
    Panel(data["input (NEURON)"].spiketrains)]
for var in ('v', 'gsyn_exc'):
    for label in ("NEST-NEST", "NEURON-NEST", "NEST-NEURON", "NEURON-NEURON"):
        panels.append(
            Panel(data[label].filter(name=var)[0], data_labels=[label]))
Figure(*panels).save("Results/music_demo.png")
