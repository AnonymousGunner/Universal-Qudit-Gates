# Minimal-Universal-Multi-Qudit-Gate-Sets
The code repository alligns with the research paper titled "Practically Implementable Minimal Universal Gate Sets for Multi-Qudit Systems with Cryptographic Validation".

## 1. Introduction

This repository provides a practical implementation and validation framework for a minimal universal gate set for multi-qudit systems. 
It aligns with the paper titled "Practically Implementable Minimal Universal Gate Sets for Multi-Qudit Systems with Cryptographic Validation". 
Unlike generic qubit circuits, qudit circuits leverage higher-dimensional Hilbert spaces. 
Here we validate two cryptographically significant algorithms: **Grover's search** and **Quantum Key Distribution (QKD)**. 
Both implementations are constructed with traditional gates and with decomposed gates synthesized entirely from the universal set `PHASE1 ∪ T_elements`. 
The aim is to confirm functional equivalence and assess performance trade-offs, thereby demonstrating that qudit-based circuits can be realistically deployed for cryptographic applications.

## 2. Technical Flow

- For **Grover’s algorithm**: two 4-dimensional qudits are initialized.  
  - Traditional circuit → Generalized Hadamard, Uf, U0, measurement.  
  - Decomposed circuit → Reck’s decomposition of Hadamard-like gates into 2×2 rotations + PHASE1 gates.  
- For **QKD**:  
  - Alice encodes states in rectilinear/diagonal basis.  
  - Bob randomly selects measurement basis.  
  - Both traditional and universal gates are used.  
  - Keys are extracted where bases match.  
- Outputs include:  
  - Histograms of Grover amplified states, one with traditional gates, another with decomposed gates.  
  - Basis choice distribution plots for Alice and Bob.  
  - Logs with per-round QKD results.  

### Cryptographic Relevance

- Grover’s algorithm validates **attack feasibility** in qudit cryptanalysis.  
- QKD demonstrates **defensive protocol correctness** under decomposition.  
- Equivalence testing confirms minimal gate sets can support cryptography securely.  


## 3. Grover’s Algorithm Validation

### Traditional Circuit

Constructed with generalized qudit gates:  

```python
circuit = cirq.Circuit([
    QuditHGate(d=4)(q0),
    QuditHGate(d=4)(q1),
    UfGate()(q0,q1),
    QuditHGate(d=4)(q0),
    QuditHGate(d=4)(q1),
    U0Gate()(q0,q1),
    QuditHGate(d=4)(q0),
    QuditHGate(d=4)(q1),
    cirq.measure(q0, q1)
])
```

### Universal Circuit

Constructed using Reck’s decomposition:  

```python
H_unitary = QuditHGate(d=4)._unitary_()
R_matrices, Phi = reckon_decompose_unitary(H_unitary)
ops = [ArbitraryGate(d=4, matrix=Phi)(q0), ArbitraryGate(d=4, matrix=Phi)(q1)]
for M in R_matrices[::-1]:
    ops += [ArbitraryGate(d=4, matrix=M)(q0), ArbitraryGate(d=4, matrix=M)(q1)]
```

### Results

- **Traditional Histogram**  
  ![Grover Traditional](./Grover_Measurements/Grovers%20Circuit%20with%20Traditional%20Gates.png)  

- **Universal Histogram**  
  ![Grover Universal](./Grover_Measurements/Grovers%20Circuit%20with%20Universal%20Gates.png)  

**Interpretation:** Both amplify the same marked state, confirming equivalence.  
The universal version has greater depth but functional correctness is preserved.  

## 4. Quantum Key Distribution (QKD) Validation

- Alice chooses random trit (0/1/2) and basis (rectilinear/diagonal).  
- Bob chooses random basis (rectilinear/diagonal).  
- Keys established when bases align.  

### Basis Choice Distribution  

![QKD Basis](./QKD_Measurements/QKD%20Basis%20Choice%20Distribution.png)  

### Example Logs  

```
--- QKD Round 29/100 ---
Alice: Preparing bit 2 in basis 'diagonal'.
Bob: Choosing basis 'diagonal'.
0(d=3): --- Qutrit_0swap2_Gate --- Qu3H --- Qu3H --- M('diagonal_measurement')
0(d=3): --- Qu3M x3 --- Qu3M x5 --- Qu3M x5 --- M('diagonal_measurement')
Bob: Measured bit 2 in basis 'diagonal'.
Bob: Measured bit 2 in basis 'diagonal'.
Alice and Bob used the same basis.
Key established for this round: 2
```

In Round 29 of the QKD simulation, within the traditional setup, Alice randomly selected the trit “2” and the diagonal basis, while Bob independently chose the same diagonal basis. On Alice’s side, this resulted in a circuit beginning with the 0swap2 gate to encode the symbol, followed by a Hadamard gate to realize the diagonal basis. Bob, matching the basis choice, appended a Hadamard gate before measurement. 
In case of universal decomposed gates, all the gates are decomposed into unitary matrices of the proposed forms and then appended subsequently replacing the traditional gates. The 0swap2 gate has been replaced with one R_ij, one Phi balance and one Phi matrices, while the Hadamard gate has been replaced with three R_ij, one Phi balance and one Phi matrices.
Both measurements yielded the value “2”, and since the bases aligned, a key bit was successfully established, exactly as predicted by the principles of QKD.


```
--- QKD Round 67/100 ---
Alice: Preparing bit 0 in basis 'rectilinear'.
Bob: Choosing basis 'diagonal'.
20 Anonymous Submission
0(d=3): --- Qu3H --- M('rectilinear_measurement')
0(d=3): --- Qu3M --- Qu3M --- Qu3M --- Qu3M --- Qu3M --- M('diagonal_measurement')
Bob: Measured bit 1 in basis 'diagonal'.
Bob: Measured bit 2 in basis 'diagonal'.
Alice and Bob used different bases or measurement result invalid.
No key bit established for this round.
```

In Round 67 of the QKD simulation, within the traditional setup, Alice randomly selected the trit “0” and the rectilinear basis, while Bob independently chose the diagonal basis. Because of Alice’s choices, this resulted in a circuit that keeps the state unaltered. Bob, matching the basis choice, appended a Hadamard gate before measurement. 
In case of universal decomposed gates, the only Hadamard gate in the circuit has been replaced with three R_ij, one Phi balance and one Phi matrices. We see both measurements yielded different values of measured bit, although the choice of bases being different, this round will anyway not yield any shared key bit.

### Summary  

- Typically 100 rounds simulated.  
- Expected that around 50 shared key bits established.  
- Both traditional and universal gates produced identical outcomes whenever choices of bases matches.  

## 5. Implementation Details

- **Reck’s decomposition** expresses arbitrary unitaries as products of 2×2 rotations and a diagonal phase.  
- Custom Cirq gates are decomposed into `ArbitraryGate` instances.  
- Circuits use `cirq.LineQid` with dimension=3 (QKD) or 4 (Grover).  
- Outputs:  
  - Histograms (`Matplotlib`)  
  - Logs (plain text)  

## 6. Results & Outputs

### Sample Grover Outputs  

- **Grovers Circuit with Traditional Gates**.  
  - Measurement Outcomes → Grover's algorithm has amplified the probability of the state S_k.  
  - Example of States with most probability → S_k - 2404. S_k1 - 204. S_k2 - 188. 
  - Histogram of states with respective probability gets stored.

- **Grovers Circuit with Universal Gates**.  
  - Measurement Outcomes → Grover's algorithm has amplified the probability of the state S_k.  
  - Example of States with most probability → S_k - 2456. S_k1 - 191. S_k2 - 190. 
  - Histogram of states with respective probability gets stored.


### Sample QKD Outputs  

- **QKD Simulation**.  
  - Total Rounds Simulated → 100  
  - Number of Key Bits Established → 56 
  - Shared Secret Key → 12201220120122220110210202122200201211121012212020010001
  - Impression → The choosen bases of Alice and Bob has matched 56 times and 56 many times the circuits with traditional and universal gates has yielded same measurement.
  - QKD Simulations logs gets printed.
  - Histogram of QKD Basis Choice Distribution gets stored.


## 7. How to Run

### Requirements  

All necessary python libraries with matching version are provided in requirements file. 
Majorly required python libraries are following. 

- Python 3.9+  
- Cirq  
- NumPy  
- Matplotlib  

### Installation  

```bash
pip install -r requirements.txt
```

### Run Simulation  

Run the provied notebook files within the python environment with requried packages. 

---

## ✅ Conclusion
This repository validates the practicality of the minimal universal gate set (`PHASE1 ∪ T_elements`) by demonstrating end-to-end cryptographic protocols in a reproducible Python framework.  
Both **Grover's Algorithm** and **QKD Simulation** confirm **functional equivalence** between traditional and decomposed implementations, ensuring security and scalability in cryptographic contexts.



