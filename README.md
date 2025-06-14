# qc-sudoku-solver

The goal of this project is to create a Qiskit function that takes a
Sudoku puzzle and ouputs a solution, using Grover's algorithm.

**Note:** I have been struggling to get Github-flavored Markdown to play
nicely with LaTeX, so some mathematical expressions below may not be
correctly displayed. I'll try to fix the issues when I have time.

## Minimum viable project

Here are my requirements for a minimum viable project.

- Constrain the possible inputs
  - $4 \times 4$ puzzles
  - Assume a solution *exists* and is *unique*
- May use only the following gates
  - Arbitrary 1-qubit gates
  - $CX$ gates
  - Toffoli gates
- Documentation
- Explanation of construction
- Benchmarking

## Extensions

If I have time after finishing my minimum viable project, I may pursue
the following extensions.

- $n^2 \times n^2$ puzzles for $n = 3,4$
- Puzzles with no solutions (should be able to determine if a solution exists)
- Puzzles with multiple solutions (need to deal with the souffle problem)

---

Here are my initial thoughts on an actual implementation.

## Setting up the problem

Suppose we have a $n^2 \times n^2$ Sudoku grid with $k$ empty cells, and
suppose it has a unique solution $`x^* \in \mathbb{Z}_{n^2}^k`$ describing
how to fill the empty cells.
We can view our problem as an unstructured search on $\mathbb{Z}_{n^2}^k$,
represented by a function

$$
f: \mathbb{Z}_{n^2}^k \to \mathbb{F}_2
$$

given by $f^{-1}(-1) = \{x^*\}$.

Suppose $n=2$, so there each cell can take on 4 possible values,
which can each be represented by 2 qubits.

$$
\lvert 00 \rangle,
\lvert 01 \rangle,
\lvert 10 \rangle,
\lvert 11 \rangle.
$$

Thus, we are searching the space $(\mathbb{F}_2^2)^k$, where candidate
solutions can be thought of as a sequence of $k$ 2-bit binary numbers

$$
x = (x_0, x_1, x_2, \dots, x_{k-1}).
$$

We may write each binary number $x_i \in \mathbb{F}_2^2$ as

$$
x_i = 2^1 x_{i1} + 2^0 x_{i0},
$$

or simply $x_i = x_{i1} x_{i0}$.

## Grover's algorithm

Suppose we have an oracle $O$ that performs the transformation

$$
O \lvert x \rangle \lvert q \rangle
= \lvert x \rangle \lvert q \oplus f(x) \rangle.
$$

Let $\lvert \psi \rangle$ denote the equal superposition state

$$
\lvert \psi \rangle
= \frac{1}{\sqrt{N}} \sum_{x \in \mathbb{F}_2^{2k}} \lvert x \rangle,
$$

where $N = 2^{2k}$.

Grover's algorithm will proceed as follows.

**Inputs:**

- oracle $O$
- $2k+1$ qubits, each in the state $\lvert 0 \rangle$

**Outputs:** $x^*$ (with high probability)

**Procedure**

1. Start with the initial state $\lvert 0 \rangle^{\otimes 2k} \lvert 0 \rangle$.
2. Apply the Hadamard transform $H^{\otimes 2k}$ to the first $2k$ qubits, and $HX$ to the last qubit.
3. Apply the Grover iteration $G = (2 \lvert \psi \rangle \langle \psi \rvert - I)O$ repeatedly, namely $R \approx \left\lceil \frac{\pi \sqrt{N}}{4} \right\rceil$ times.
4. Measure the first $2k$ qubits to obtain $x^*$.

Here are the states as we go through each step.

$$
\begin{align}
|0\rangle^{\otimes 2k} \lvert 0 \rangle
&\xrightarrow{\text{Step 2}}
\lvert \psi \rangle \lvert - \rangle \\
&\xrightarrow{\text{Step 3}}
G^R \lvert \psi \rangle \lvert - \rangle
\approx \lvert x^* \rangle \lvert - \rangle \\
&\xrightarrow{\text{Step 4}}
x^*
\end{align}
$$

## Considerations

We need to figure out the following details.

- Represent the initial Sudoku puzzle. (Choose a data structure.)
- Extract the empty cells as input for Grover's algorithm.
- Implement the oracle $O$ so that it checks the rules of Sudoku.
