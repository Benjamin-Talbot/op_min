from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_algorithms.utils import algorithm_globals

algorithm_globals.random_seed = 42

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

backend = AerSimulator(device='GPU')

job = backend.run(qc, shots=1000)
result = job.result()
counts = result.get_counts()
print(counts)
