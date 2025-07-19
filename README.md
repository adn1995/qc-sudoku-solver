# qc-sudoku-solver

The goal of this project is to create a Qiskit function that uses Grover's
algorithm to solve a sudoku puzzle.

For this project, I wrote Qiskit functions that construct a quantum circuit
that solves an $n^2 \times n^2$ sudoku puzzle.
I also wrote a notebook that explains these circuits, using a $4 \times 4$
puzzle as an example.

I followed the following constraints and assumptions when writing these
circuits.

- Assume a solution *exists* and is *unique*
- May use only the following gates
  - Arbitrary 1-qubit gates
  - $CX$ gates
  - Toffoli gates
  - Multicontrolled $X$ or $Z$ gates, only for the marker oracle

## Relevant files

This repo has two important files.

- [`qc_sudoku.py`](https://github.com/adn1995/qc-sudoku-solver/blob/main/qc_sudoku.py) contains the source code for my circuits
- [`qc-sudoku-solver.ipynb`](https://github.com/adn1995/qc-sudoku-solver/blob/main/qc-sudoku-solver.ipynb) explains the circuits, using a $4 \times 4$ puzzle as an example

In the Jupyter notebook `qc-sudoku-solver.ipynb`, you will find explanations of
my implementations of various subroutines.

- Marker oracle
- Grover diffuser
- Subcircuit for checking if two sudoku cells are equal
- Subcircuit for checking if a "group" (row, column, or block) satisfies the rules of sudoku

You will also find explanations of how these subroutines are combined in a
quantum circuit that takes an unsolved puzzle and outputs a quantum state
representing a valid solution of the puzzle.

## References

I relied on *Quantum Computation and Quantum Information* by Nielsen and Chuang
to help me understand Grover's algorithm.
