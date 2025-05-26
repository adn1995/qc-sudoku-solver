# qc-sudoku-solver

The goal of this project is to create a Qiskit function that takes a
Sudoku puzzle and ouputs a solution, using Grover's algorithm.

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
