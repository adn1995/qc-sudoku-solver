# filename: qc_sudoku.py
# author: Arthur Diep-Nguyen

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.quantum_info import Statevector
#import matplotlib.pyplot as plt

import numpy as np
import math

########################################################################
########################################################################
# Main function for solving sudoku puzzles
########################################################################
########################################################################

def solve(puzzle: np.array) -> np.array:
    """Solves a sudoku puzzle using Grover's algorithm.

    This function assumes the input is an n^2 by n^2 sudoku puzzle, with
    entries in the range [0, n^2-1], with a unique solution.

    #TODO Rewrite to deal with zero or multiple solutions

    Parameters
    ----------
    puzzle : np.array
        n^2 by n^2 array, representing the unsolved puzzle,
        where an empty cell is np.nan

    Returns
    -------
    np.array
        n^2 by n^2 array, representing the solved puzzle

    Examples
    --------
    >>> puzzle = np.array([[2,0,3,1],[1,np.nan,np.nan,0],
    ...                   [0,np.nan,np.nan,np.nan],[3,np.nan,np.nan,2]])
    >>> solve(puzzle)
    array([[2, 0, 3, 1],
           [1, 3, 2 ,0],
           [0, 2, 1, 3],
           [3, 1, 0, 2]])
    """
    # Throw out garbage inputs
    if type(puzzle) != np.array:
        raise TypeError("Expected a numpy array")
    elif not possibly_sudoku(puzzle):
        raise ValueError("Not a valid sudoku")

    # Important constants
    num_qubits_per_entry = find_num_qubits_per_entry(puzzle)
    num_unknown = find_num_unknown(puzzle)
    num_qubits = num_qubits_per_entry * num_unknown
    niter = math.ceil(math.pi * math.sqrt(2**num_qubits) / 4)

    # Run Grover circuit and get output bitstring
    grover_circuit = grover(puzzle, niter)
    output_bitstring = most_likely_state(grover_circuit,
        grover_circuit.qregs[0])
    assert len(output_bitstring) == num_qubits

    # Convert to list of integers
    entry_bitstrings = [output_bitstring[i:i+num_qubits_per_entry]
        for i in range(num_unknown)]
    entry_ints = [bitstring_to_int(x) for x in entry_bitstrings]

    # Use output bitstring to fill in the unsolved puzzle
    solution = puzzle.copy()
    for idx, x in np.ndenumerate(puzzle):
        if math.isnan(x):
            solution[idx[0], idx[1]] = entry_ints.pop(0)
    return solution

########################################################################
########################################################################
# Non-quantum helper functions
########################################################################
#########################################################################

def possibly_sudoku(arr: np.array) -> bool:
    """Returns whether or not the given array is a sudoku puzzle, maybe.

    This function checks a few basic facts about the given array.
    - The array should be a square matrix
    - The dimension should be a square number n^2
    - The entries should be in the range [0, n^2-1], or np.nan

    This function returns True if all of the above facts are True, and
    False otherwise.
    There may be false positives, but no false negatives.

    This function does not check all the rules of sudoku, but instead
    simply helps to prevent garbage inputs.

    Parameters
    ----------
    arr : np.array

    Returns
    -------
    bool

    Examples
    --------
    Valid solved sudoku returns True

    >>> array1 = np.array([[2,0,3,1],[1,3,2,0],[0,2,1,3],[3,1,0,2]])
    >>> possibly_sudoku(array1)
    True

    Valid unsolved sudoku returns True

    >>> array2 = np.array([[2,0,3,1], [1,np.nan,np.nan,0],
    ...                   [0,np.nan,np.nan,np.nan],[3,np.nan,np.nan,2]])
    >>> possibly_sudoku(array2)
    True

    Non-square array returns False

    >>> array3 = np.array([[0,1,2],[0,1,2]])
    >>> possibly_sudoku(array3)
    False

    Square array with non-square dimensions returns False

    >>> array4 = np.array([[0,1,2],[1,2,0],[2,1,0]])
    >>> possibly_sudoku(array4)
    False

    4x4 array with invalid entries returns False

    >>> array5 = np.array([[1,2,3,4],[5,6,7,8],
    ...                   [9,10,11,12],[13,14,15,16]])
    >>> possibly_sudoku(array5)
    False

    Example of a false positive

    >>> array6 = np.zeros((4,4))
    >>> possibly_sudoku(array6)
    True
    """
    if not (is_square_matrix(arr) and is_square_dim(arr)):
        return False
    else:
        num_rows = arr.shape[0]
        valid_entries = {x for x in range(num_rows)}
        for entry in np.nditer(arr):
            if entry not in valid_entries:
                return False
        return True

def is_square_matrix(arr: np.array) -> bool:
    size = arr.size
    num_rows = math.isqrt(size)
    return (np.shape == (num_rows,num_rows))

def is_square_dim(arr: np.array) -> bool:
    num_rows = arr.shape[0]
    return (num_rows == math.isqrt(num_rows)**2)

def find_num_qubits_per_entry(puzzle: np.array) -> int:
    """Returns the number of (qu)bits needed to represent an entry.

    Examples
    --------
    In a 4x4 puzzle, the possible values for an entry are {0,1,2,3},
    each of which can be represented with 2 (qu)bits.

    >>> puzzle = np.array([[2,0,3,1], [1,np.nan,np.nan,0],
    ...                   [0,np.nan,np.nan,np.nan],[3,np.nan,np.nan,2]])
    >>> num_qubits_per_entry(puzzle)
    2
    """
    num_rows = puzzle.shape[0]
    return math.ceil(math.log2(num_rows))

def find_num_unknown(puzzle: np.array) -> int:
    """Returns the number of unknown (i.e. empty) entries in the puzzle.

    Examples
    --------
    >>> puzzle = np.array([[2,0,3,1], [1,np.nan,np.nan,0],
    ...                   [0,np.nan,np.nan,np.nan],[3,np.nan,np.nan,2]])
    >>> num_unknown(puzzle)
    7
    """
    return np.count_nonzero(np.isnan(puzzle))

def find_num_known(puzzle: np.array) -> int:
    """Returns the number of known (i.e. filled) entries in the puzzle.

    Examples
    --------
    >>> puzzle = np.array([[2,0,3,1], [1,np.nan,np.nan,0],
    ...                   [0,np.nan,np.nan,np.nan],[3,np.nan,np.nan,2]])
    >>> num_known(puzzle)
    9
    """
    return np.count_nonzero(~np.isnan(puzzle))

def bitstring_to_int(bitstring: str) -> int:
    """Returns the integer corresponding to a bitstring.

    Examples
    --------
    >>> bitstring_to_int("101010")
    42

    >>> bitstring_to_int("1000101")
    69

    >>> bitstring_to_int("110100100")
    420

    >>> bitstring_to_int("0000000110100100")
    420
    """
    return int(bitstring, base=2)

def int_to_bitstring(value: int, num_bits: int) -> str:
    """Returns the bitstring corresponding to an integer, padding with
    zeros to get the desired number of bits.

    Examples
    --------
    >>> int_to_bitstring(42, 6)
    '101010'

    >>> int_to_bitstring(42, 8)
    '00101010'
    """
    bitstring = bin(value)[2:]
    diff = num_bits - len(bitstring)
    if diff < 0:
        raise ValueError("`num_bits` is too low to represent `value`")
    return bitstring.zfill(num_bits)

def find_rows_with_nan(puzzle: np.array) -> set:
    """Returns the set of row indices for rows with empty cells.

    Examples
    --------
    >>> puzzle = np.array([[2,0,3,1], [1,np.nan,np.nan,0],
    ...                   [0,np.nan,np.nan,np.nan],[3,np.nan,np.nan,2]])
    >>> find_rows_with_nan(puzzle)
    (1, 2, 3)
    """
    rows_with_nan = set()
    index = 0
    for row in puzzle:
        if np.isnan(np.sum(row)):
            rows_with_nan.add(index)
        index += 1
    assert index == puzzle.shape[0]
    return rows_with_nan

def find_cols_with_nan(puzzle: np.array) -> set:
    """Returns the set of column indices for columns with empty cells.

    Examples
    --------
    >>> puzzle = np.array([[2,0,3,1], [1,np.nan,np.nan,0],
    ...                   [0,np.nan,np.nan,np.nan],[3,np.nan,np.nan,2]])
    >>> find_cols_with_nan(puzzle)
    (1, 2, 3)
    """
    cols_with_nan = set()
    index = 0
    for col in np.transpose(puzzle):
        if np.isnan(np.sum(col)):
            cols_with_nan.add(index)
        index += 1
    assert index == puzzle.shape[1]
    return cols_with_nan

def find_blocks_with_nan(puzzle: np.array) -> set:
    """Returns the set of block indices for blocks with empty cells.

    Examples
    --------
    In a 4x4 grid, we have 4 blocks, which we will index as
        [0 1]
        [2 3]

    >>> puzzle = np.array([[2,0,3,1], [1,np.nan,np.nan,0],
    ...                   [0,np.nan,np.nan,np.nan],[3,np.nan,np.nan,2]])
    >>> find_blocks_with_nan(puzzle)
    (0, 1, 2, 3)
    """
    blocks_with_nan = set()
    N = puzzle.shape[0]
    n = math.isqrt(N)
    for i in range(N):
        start_row = max([n*j for j in range(n) if i >= n*j])
        start_col = n*(i-start_row)
        grid = puzzle[start_row:start_row+n, start_col:start_col+n]
        if np.isnan(np.sum(grid)):
            blocks_with_nan.add(i)
    return blocks_with_nan

########################################################################
########################################################################
# Quantum helper functions
########################################################################
########################################################################

def most_likely_state(qc: QuantumCircuit, qr: QuantumRegister) -> str:
    """Returns the binary representation of the most likely state.

    Given a quantum circuit `qc` and quantum register `qr`, this
    function returns the binary representation of the computational
    basis vector that is most likely to be measured from the given
    register.

    Parameters
    ----------
    qc : QuantumCircuit
    qr : QuantumRegister

    Returns
    -------
    str
        Binary representation of the most likely basis vector as a str

    Examples
    --------
    #TODO
    """
    output_state = Statevector(qc)
    prob_dict = output_state.probability_dict(qr)
    return max(prob_dict, key=prob_dict.get)

########################################################################
# Basic gates and circuits
########################################################################

def AND_gate() -> Gate:
    """Returns a quantum gate corresponding to logical AND.

    Just the Toffoli gate.
    """
    input_qr = QuantumRegister(2, name="in")
    target_qr = AncillaRegister(1, name="out")
    qc = QuantumCircuit(input_qr, target_qr, name="AND")
    qc.ccx(input_qr[0], input_qr[1], target_qr[0])
    return qc.to_gate()

def NAND_gate() -> Gate:
    """Returns a quantum gate corresponding to logical NAND.

    Toffoli gate with an X gate at the end.
    """
    input_qr = QuantumRegister(2, name="in")
    target_qr = AncillaRegister(1, name="out")
    qc = QuantumCircuit(input_qr, target_qr, name="NAND")
    qc.ccx(input_qr[0], input_qr[1], target_qr[0])
    qc.x(target_qr[0])
    return qc.to_gate()

def XOR_gate() -> Gate:
    """Returns a quantum gate corresponding to logical XOR.
    """
    input_qr = QuantumRegister(2, name="in")
    target_qr = AncillaRegister(1, name="out")
    qc = QuantumCircuit(input_qr, target_qr, name="XOR")
    qc.cx(input_qr[0], target_qr[0])
    qc.cx(input_qr[1], target_qr[0])
    return qc.to_gate()

def XNOR_gate() -> Gate:
    """Returns a quantum gate corresponding to logical XOR.

    XOR with an X gate at the end.
    """
    input_qr = QuantumRegister(2, name="in")
    target_qr = AncillaRegister(1, name="out")
    qc = QuantumCircuit(input_qr, target_qr, name="XNOR")
    qc.cx(input_qr[0], target_qr[0])
    qc.cx(input_qr[1], target_qr[0])
    qc.x(target_qr[0])
    return qc.to_gate()

def OR_gate() -> Gate:
    """Returns a quantum gate corresponding to logical OR.

    Compose AND and XOR.
    """
    input_qr = QuantumRegister(2, name="in")
    target_qr = AncillaRegister(1, name="out")
    qc = QuantumCircuit(input_qr, target_qr, name="OR")
    qc.ccx(input_qr[0], input_qr[1], target_qr[0])
    qc.cx(input_qr[0], target_qr[0])
    qc.cx(input_qr[1], target_qr[0])
    return qc.to_gate()

def NOR_gate() -> Gate:
    """Returns a quantum gate corresponding to logical NOR.

    OR with an X gate at the end.
    """
    input_qr = QuantumRegister(2, name="in")
    target_qr = AncillaRegister(1, name="out")
    qc = QuantumCircuit(input_qr, target_qr, name="OR")
    qc.ccx(input_qr[0], input_qr[1], target_qr[0])
    qc.cx(input_qr[0], target_qr[0])
    qc.cx(input_qr[1], target_qr[0])
    qc.x(target_qr[0])
    return qc.to_gate()

def value_to_ancilla(value: int, num_qubits: int) -> Gate:
    """Returns a quantum gate that stores the given value in the ancilla
    register.
    """
    bitstring = int_to_bitstring(value, num_qubits)
    qr = AncillaRegister(len(bitstring), name="a")
    qc = QuantumCircuit(qr, name="Set value")
    qc.x([qr[i] for i in range(len(bitstring)) if bitstring[i] == "1"])

    return qc.to_gate()

def prepare_known_ancilla(puzzle: np.array) -> Gate:
    """Returns a quantum gate that prepares the state of the ancilla
    registers reserved for storing the values of known values.

    The initial sudoku has some empty (unknown) cells and some filled
    (known) cells.
    In the `grover` function, the known cells are assigned to ancilla
    registers.
    This function stores the value of the known cells in the
    corresponding ancilla registers.

    Parameters
    ----------
    puzzle : np.array
        n^2 by n^2 array, representing the unsolved puzzle,
        where an empty cell is np.nan

    Returns
    -------
    Gate
    """
    num_known = find_num_known(puzzle)
    num_qubits_per_entry = find_num_qubits_per_entry(puzzle)
    num_qubits = num_known * num_qubits_per_entry
    qr = AncillaRegister(num_qubits, name="a")
    qc = QuantumCircuit(qr, name="Prepare known values")

    index = 0
    for x in np.nditer(puzzle):
        if not np.isnan(x):
            qc.compose(value_to_ancilla(x, num_qubits_per_entry),
                qc[index:index+num_qubits_per_entry],
                inplace=True)
            index += num_qubits_per_entry
    assert index == num_qubits

    return qc.to_gate()

########################################################################
# Gates and circuits for Grover's algorithm (besides the oracle)
#
# The oracle is complicated, so I'll give the oracle its own section.
########################################################################

def grover(puzzle: np.array, niter: int) -> QuantumCircuit:
    """Returns the Grover circuit for the given puzzle.

    This function assumes the input is an n^2 by n^2 sudoku puzzle, with
    entries in the range [0, n^2-1], with a unique solution.
    The output is the corresponding quantum circuit implementing
    Grover's algorithm, using `niter` Grover iterations.

    #TODO Rewrite to deal with zero or multiple solutions

    Parameters
    ----------
    puzzle : np.array
        n^2 by n^2 array, representing the unsolved puzzle,
        where an empty cell is np.nan
    niter : int
        Number of Grover iterations in the circuit

    Returns
    -------
    QuantumCircuit
        Quantum circuit, implementing Grover's algorithm for the given
        sudoku puzzle

    Examples
    --------
    #TODO
    """
    # Important constants
    num_unknown = find_num_unknown(puzzle)
    num_known = find_num_known(puzzle)
    num_qubits_per_entry = find_num_qubits_per_entry(puzzle)

    rows_with_nan = find_rows_with_nan(puzzle)
    cols_with_nan = find_cols_with_nan(puzzle)
    blocks_with_nan = find_blocks_with_nan(puzzle)

    # Each unknown cell gets enough qubits
    unknown_regs = [QuantumRegister(num_qubits_per_entry,
        name="unknown{}".format(i))
        for i in range(num_unknown)]

    # Each known cell gets enough qubits
    known_regs = [AncillaRegister(num_qubits_per_entry,
        name="known{}".format(i))
        for i in range(num_known)]

    # Registers that flip if a sudoku rule is satisfied
    row_regs = [AncillaRegister(1, name="row{}".format(row))
        for row in rows_with_nan]
    col_regs = [AncillaRegister(1, name="col{}".format(col))
        for col in cols_with_nan]
    block_regs = [AncillaRegister(1, name="block{}".format(block))
        for block in blocks_with_nan]

    # Oracle qubit, which should flip if all rules are satisfied
    oracle_qubit = AncillaRegister(1, name="q")

    # Need some ancilla qubits for doing work
    # ancilla = AncillaRegister()

    #TODO prepare state with hadamard transform
    #TODO loop for Grover iteration
    pass

def grover_iteration(puzzle: np.array) -> Gate:
    """Returns the Grover iteration circuit for the given puzzle.

    This function assumes the input is an n^2 by n^2 sudoku puzzle, with
    entries in the range [0, n^2-1], with a unique solution.
    The output is the corresponding quantum circuit implementing the
    Grover iteration, consisting of the oracle and diffuser.

    #TODO Rewrite to deal with zero or multiple solutions

    Parameters
    ----------
    puzzle : np.array
        n^2 by n^2 array, representing the unsolved puzzle,
        where an empty cell is np.nan

    Returns
    -------
    Gate
        Quantum gate, implementing the Grover iteration for the given
        sudoku puzzle

    Examples
    --------
    #TODO
    """
    #TODO need to know how many registers I need
    #TODO marker oracle
    #TODO grover diffuser
    pass

def grover_diffuser(num_qubits: int) -> Gate:
    """Returns the Grover diffuser circuit on `num_qubits` qubits.
    """
    qr = QuantumRegister(num_qubits, name="x")
    qc = QuantumCircuit(qr, name="Diffuser")

    qc.h(qr)
    qc.x(qr)

    # Not sure if there is a method like qc.mcz()
    # Multi-controlled Z is equivalent to the following
    qc.h(qr[-1])
    qc.mcx(qr[:-1], qr[-1])
    qc.h(qr[-1])

    qc.x(qr)
    qc.h(qr)

    return qc.to_gate()

########################################################################
# The marker oracle
########################################################################

def oracle(puzzle: np.array) -> Gate:
    """Returns the marker oracle for the given puzzle.

    This function assumes the input is an n^2 by n^2 sudoku puzzle, with
    entries in the range [0, n^2-1], with a unique solution.
    The output is the corresponding quantum circuit implementing the
    marker oracle, which flips the oracle qubit if and only if it sees a
    valid solution to the puzzle.

    #TODO Rewrite to deal with zero or multiple solutions

    Parameters
    ----------
    puzzle : np.array
        n^2 by n^2 array, representing the unsolved puzzle,
        where an empty cell is np.nan

    Returns
    -------
    Gate
        Quantum gate, implementing the marker oracle for the given
        sudoku puzzle

    Examples
    --------
    #TODO
    """
    #TODO unknown_register
    #TODO known_register
    #TODO oracle_register
    pass
