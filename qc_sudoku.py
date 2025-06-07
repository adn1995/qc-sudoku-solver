# filename: qc_sudoku.py
# author: Arthur Diep-Nguyen

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.quantum_info import Statevector
#import matplotlib.pyplot as plt

import numpy as np
import math
from itertools import combinations

########################################################################
########################################################################
# Main function for solving sudoku puzzles
########################################################################
########################################################################

def solve(puzzle: np.ndarray) -> np.ndarray:
    """Solves a sudoku puzzle using Grover's algorithm.

    This function assumes the input is an n^2 by n^2 sudoku puzzle, with
    entries in the range [0, n^2-1], with a unique solution.

    #TODO Rewrite to deal with zero or multiple solutions

    Parameters
    ----------
    puzzle : np.ndarray
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
    if type(puzzle) != np.ndarray:
        raise TypeError("Expected a numpy array")
    elif not possibly_sudoku(puzzle):
        raise ValueError("Not a valid sudoku")

    # Important constants
    nqubits_per_entry = find_nqubits_per_entry(puzzle)
    num_unknown = find_num_unknown(puzzle)
    nqubits = nqubits_per_entry * num_unknown
    niter = math.ceil(math.pi * math.sqrt(2**nqubits) / 4)

    # Run Grover circuit and get output bitstring
    grover_circuit = grover(puzzle, niter)

    # First `nqubit` registers should correspond to the empty cells
    output_bitstring = most_likely_state(grover_circuit,
                                            [i for i in range(nqubits)])
    assert len(output_bitstring) == nqubits

    # Convert to list of integers
    entry_bitstrings = [output_bitstring[i:i+nqubits_per_entry]
                        for i in range(num_unknown)]
    entry_ints = [bitstr_to_int(x) for x in entry_bitstrings]

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

def possibly_sudoku(arr: np.ndarray) -> bool:
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
    if not (is_sq_matrix(arr) and is_sq_dim(arr)):
        return False
    else:
        nrows = arr.shape[0]
        valid_entries = {x for x in range(nrows)}
        for entry in np.nditer(arr):
            if entry not in valid_entries:
                return False
        return True

def is_sq_matrix(arr: np.ndarray) -> bool:
    """Returns if an array is a square matrix.
    """
    size = arr.size
    nrows = math.isqrt(size)
    return (np.shape == (nrows,nrows))

def is_sq_dim(arr: np.ndarray) -> bool:
    """Returns if an array (presumably a square matrix) has a square
    number of rows.
    """
    nrows = arr.shape[0]
    return (nrows == math.isqrt(nrows)**2)

def find_nqubits_per_entry(puzzle: np.ndarray) -> int:
    """Returns the number of (qu)bits needed to represent an entry.

    Examples
    --------
    In a 4x4 puzzle, the possible values for an entry are {0,1,2,3},
    each of which can be represented with 2 (qu)bits.

    >>> puzzle = np.array([[2,0,3,1], [1,np.nan,np.nan,0],
    ...                   [0,np.nan,np.nan,np.nan],[3,np.nan,np.nan,2]])
    >>> nqubits_per_entry(puzzle)
    2
    """
    nrows = puzzle.shape[0]
    return math.ceil(math.log2(nrows))

def find_num_unknown(puzzle: np.ndarray) -> int:
    """Returns the number of unknown (i.e. empty) entries in the puzzle.

    Examples
    --------
    >>> puzzle = np.array([[2,0,3,1], [1,np.nan,np.nan,0],
    ...                   [0,np.nan,np.nan,np.nan],[3,np.nan,np.nan,2]])
    >>> num_unknown(puzzle)
    7
    """
    return np.count_nonzero(np.isnan(puzzle))

def find_num_known(puzzle: np.ndarray) -> int:
    """Returns the number of known (i.e. filled) entries in the puzzle.

    Examples
    --------
    >>> puzzle = np.array([[2,0,3,1], [1,np.nan,np.nan,0],
    ...                   [0,np.nan,np.nan,np.nan],[3,np.nan,np.nan,2]])
    >>> num_known(puzzle)
    9
    """
    return np.count_nonzero(~np.isnan(puzzle))

def bitstr_to_int(bitstr: str) -> int:
    """Returns the integer corresponding to a bitstring.

    Examples
    --------
    >>> bitstr_to_int("101010")
    42

    >>> bitstr_to_int("1000101")
    69

    >>> bitstr_to_int("110100100")
    420

    >>> bitstr_to_int("0000000110100100")
    420
    """
    return int(bitstr, base=2)

def int_to_bitstr(value: int, nbits: int) -> str:
    """Returns the bitstring corresponding to an integer, padding with
    zeros to get the desired number of bits.

    Examples
    --------
    >>> int_to_bitstr(42, 6)
    '101010'

    >>> int_to_bitstr(42, 8)
    '00101010'
    """
    bitstr = bin(value)[2:]
    diff = nbits - len(bitstr)
    if diff < 0:
        raise ValueError("`num_bits` is too low to represent `value`")
    return bitstr.zfill(nbits)

def find_rows_with_nan(puzzle: np.ndarray) -> tuple:
    """Returns the tuple of row indices for rows with empty cells.

    Examples
    --------
    >>> puzzle = np.array([[2,0,3,1], [1,np.nan,np.nan,0],
    ...                   [0,np.nan,np.nan,np.nan],[3,np.nan,np.nan,2]])
    >>> find_rows_with_nan(puzzle)
    (1, 2, 3)
    """
    rows_with_nan = []
    index = 0
    for row in puzzle:
        if np.isnan(np.sum(row)):
            rows_with_nan.append(index)
        index += 1
    assert index == puzzle.shape[0]
    return tuple(rows_with_nan)

def find_cols_with_nan(puzzle: np.ndarray) -> tuple:
    """Returns the tuple of column indices for columns with empty cells.

    Examples
    --------
    >>> puzzle = np.array([[2,0,3,1], [1,np.nan,np.nan,0],
    ...                   [0,np.nan,np.nan,np.nan],[3,np.nan,np.nan,2]])
    >>> find_cols_with_nan(puzzle)
    (1, 2, 3)
    """
    cols_with_nan = []
    index = 0
    for col in np.transpose(puzzle):
        if np.isnan(np.sum(col)):
            cols_with_nan.append(index)
        index += 1
    assert index == puzzle.shape[1]
    return tuple(cols_with_nan)

def find_blocks_with_nan(puzzle: np.ndarray) -> tuple:
    """Returns the tuple of block indices for blocks with empty cells.

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
    blocks_with_nan = []
    N = puzzle.shape[0]
    n = math.isqrt(N)
    for i in range(N):
        start_row = max([n*j for j in range(n) if i >= n*j])
        start_col = n*(i-start_row)
        grid = puzzle[start_row:start_row+n, start_col:start_col+n]
        if np.isnan(np.sum(grid)):
            blocks_with_nan.append(i)
    return tuple(blocks_with_nan)

########################################################################
########################################################################
# Quantum helper functions
########################################################################
########################################################################

def most_likely_state(qc: QuantumCircuit, qargs: list) -> str:
    """Returns the binary representation of the most likely state.

    Given a quantum circuit `qc` and list `qargs` of qubits, this
    function returns the binary representation of the computational
    basis vector that is most likely to be measured from the given
    qubits.

    Parameters
    ----------
    qc : QuantumCircuit
    qargs : list
        List of qubits to be "measured."

    Returns
    -------
    str
        Binary representation of the most likely basis vector as a str

    Examples
    --------
    #TODO
    """
    output_state = Statevector(qc)
    prob_dict = output_state.probabilities_dict(qargs)
    return max(prob_dict, key=prob_dict.get)

def value_to_ancilla(value: int, nqubits: int) -> QuantumCircuit:
    """Returns a quantum circuit that stores the given value in the
    ancilla register.
    """
    bitstr = int_to_bitstr(value, nqubits)
    qr = AncillaRegister(len(bitstr), name="a")
    qc = QuantumCircuit(qr, name="Set value")
    qc.x([qr[i] for i in range(len(bitstr)) if bitstr[i] == "1"])

    return qc

def prepare_known_ancilla(puzzle: np.ndarray) -> QuantumCircuit:
    """Returns a quantum circuit that prepares the state of the ancilla
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
    QuantumCircuit
    """
    num_known = find_num_known(puzzle)
    nqubits_per_entry = find_nqubits_per_entry(puzzle)
    nqubits = num_known * nqubits_per_entry
    qr = QuantumRegister(nqubits, name="a")
    qc = QuantumCircuit(qr, name="Prepare known values")

    index = 0
    for x in np.nditer(puzzle):
        if not np.isnan(x):
            qc.compose(value_to_ancilla(int(x), nqubits_per_entry),
                        qc[index:index+nqubits_per_entry],
                        inplace=True)
            index += nqubits_per_entry
    assert index == nqubits

    return qc

def is_equal(nqubits: int) -> QuantumCircuit:
    """Returns a quantum circuit that flips if two registers have the
    same state.

    Parameters
    ----------
    nqubits : int
        Size of each of the two registers

    Returns
    -------
    QuantumCircuit
    """
    in_qr1 = QuantumRegister(nqubits, name="x")
    in_qr2 = QuantumRegister(nqubits, name="y")
    out_qr = AncillaRegister(1, name="out")

    # We need to do `nqubit` comparisons, so we need that many qubits
    ancilla = AncillaRegister(nqubits, name="a")

    qc = QuantumCircuit(in_qr1, in_qr2, out_qr, ancilla)

    # Store comparisons in ancilla
    for i in range(nqubits):
        qc.compose(XOR_gate(),
                    qubits=[in_qr1[i], in_qr2[i], ancilla[i]],
                    inplace=True)

    # Flip target_qr iff all ancilla are 1
    # We are allowed to use multi-controlled X because this function
    # is used only in the marker oracle
    qc.mcx(ancilla, out_qr)

    # Uncompute ancilla
    for i in range(nqubits):
        qc.compose(XOR_gate(),
                    qubits=[in_qr1[i], in_qr2[i], ancilla[i]],
                    inplace=True)

    return qc

def make_unknown_dict(puzzle: np.ndarray) -> dict:
    """Returns a dictionary mapping empty cells to quantum registers.

    Parameters
    ----------
    puzzle : np.ndarray
        n^2 by n^2 array, representing the unsolved puzzle,
        where an empty cell is np.nan

    Returns
    -------
    dict
        Dictionary whose keys are the coordinates of unknown cells,
        where the corresponding values are quantum registers for storing
        the state of those cells
    """
    nrows = puzzle.shape[0]
    nqubits_per_entry = find_nqubits_per_entry(puzzle)
    return {(i,j) : QuantumRegister(nqubits_per_entry,
                                    name="unknown_({},{})".format(str(i),str(j)))
            for i in range(nrows) for j in range(nrows)
            if np.isnan(puzzle)[i,j]}

def make_known_dict(puzzle: np.ndarray) -> dict:
    """Returns a dictionary mapping filled cells to ancilla registers.

    Parameters
    ----------
    puzzle : np.ndarray
        n^2 by n^2 array, representing the unsolved puzzle,
        where an empty cell is np.nan

    Returns
    -------
    dict
        Dictionary whose keys are the coordinates of known cells,
        where the corresponding values are ancilla registers for storing
        the state of those cells
    """
    nrows = puzzle.shape[0]
    nqubits_per_entry = find_nqubits_per_entry(puzzle)
    return {(i,j) : AncillaRegister(nqubits_per_entry,
                                    name="unknown_({},{})".format(str(i),str(j)))
            for i in range(nrows) for j in range(nrows)
            if not np.isnan(puzzle)[i,j]}

def is_valid_grouping(nqubits: int) -> QuantumCircuit:
    """Returns a quantum circuit that checks if the grouping is valid.

    Let's call a row, column, or block, a "grouping" of cells.
    Given a grouping, this function flips an output register iff all
    cells in the grouping are distinct.

    Parameters
    ----------
    nqubits : int
        Size of a register representing a cell

    Returns
    -------
    QuantumCircuit
    """
    pass

########################################################################
# Logic gates
########################################################################

# Might as well just use Toffoli gate
# def AND_gate() -> Gate:
#     """Returns a quantum gate corresponding to logical AND.

#     Just the Toffoli gate.
#     """
#     input_qr = QuantumRegister(2, name="in")
#     target_qr = AncillaRegister(1, name="out")
#     qc = QuantumCircuit(input_qr, target_qr, name="AND")
#     qc.ccx(input_qr[0], input_qr[1], target_qr[0])
#     return qc.to_gate()

def NAND_qc() -> QuantumCircuit:
    """Returns a quantum circuit corresponding to logical NAND.

    Toffoli gate with an X gate at the end.
    """
    in_qr = QuantumRegister(2, name="in")
    out_qr = AncillaRegister(1, name="out")
    qc = QuantumCircuit(in_qr, out_qr, name="NAND")
    qc.ccx(in_qr[0], in_qr[1], out_qr[0])
    qc.x(out_qr[0])
    return qc

def XOR_qc() -> QuantumCircuit:
    """Returns a quantum circuit corresponding to logical XOR.
    """
    in_qr = QuantumRegister(2, name="in")
    out_qr = AncillaRegister(1, name="out")
    qc = QuantumCircuit(in_qr, out_qr, name="XOR")
    qc.cx(in_qr[0], out_qr[0])
    qc.cx(in_qr[1], out_qr[0])
    return qc

def XNOR_qc() -> QuantumCircuit:
    """Returns a quantum circuit corresponding to logical XOR.

    XOR with an X gate at the end.
    """
    in_qr = QuantumRegister(2, name="in")
    out_qr = AncillaRegister(1, name="out")
    qc = QuantumCircuit(in_qr, out_qr, name="XNOR")
    qc.cx(in_qr[0], out_qr[0])
    qc.cx(in_qr[1], out_qr[0])
    qc.x(out_qr[0])
    return qc

def OR_qc() -> QuantumCircuit:
    """Returns a quantum circuit corresponding to logical OR.

    Compose AND and XOR.
    """
    in_qr = QuantumRegister(2, name="in")
    out_qr = AncillaRegister(1, name="out")
    qc = QuantumCircuit(in_qr, out_qr, name="OR")
    qc.ccx(in_qr[0], in_qr[1], out_qr[0])
    qc.cx(in_qr[0], out_qr[0])
    qc.cx(in_qr[1], out_qr[0])
    return qc

def NOR_qc() -> QuantumCircuit:
    """Returns a quantum circuit corresponding to logical NOR.

    OR with an X gate at the end.
    """
    input_qr = QuantumRegister(2, name="in")
    target_qr = AncillaRegister(1, name="out")
    qc = QuantumCircuit(input_qr, target_qr, name="OR")
    qc.ccx(input_qr[0], input_qr[1], target_qr[0])
    qc.cx(input_qr[0], target_qr[0])
    qc.cx(input_qr[1], target_qr[0])
    qc.x(target_qr[0])
    return qc

########################################################################
# Gates and circuits for Grover's algorithm (besides the oracle)
#
# The oracle is complicated, so I'll give the oracle its own section.
########################################################################

def grover(puzzle: np.ndarray, niter: int) -> QuantumCircuit:
    """Returns the Grover circuit for the given puzzle.

    This function assumes the input is an n^2 by n^2 sudoku puzzle, with
    entries in the range [0, n^2-1], with a unique solution.
    The output is the corresponding quantum circuit implementing
    Grover's algorithm, using `niter` Grover iterations.

    #TODO Rewrite to deal with zero or multiple solutions

    Parameters
    ----------
    puzzle : np.ndarray
        n^2 by n^2 array, representing the unsolved puzzle,
        where an empty cell is np.nan
    niter : int
        Number of Grover iterations in the circuit

    Returns
    -------
    QuantumCircuit
        Implements Grover's algorithm for the given sudoku puzzle

    Examples
    --------
    #TODO
    """
    # Important constants
    nrows = puzzle.shape[0]
    #num_unknown = find_num_unknown(puzzle)
    #num_known = find_num_known(puzzle)
    nqubits_per_entry = find_nqubits_per_entry(puzzle)

    # Dictionaries with quantum registers for each cell in the puzzle
    unknown_dict = make_unknown_dict(puzzle)
    known_dict = make_known_dict(puzzle)
    #cell_to_regs = unknown_dict | known_dict

    rows_with_nan = find_rows_with_nan(puzzle)
    cols_with_nan = find_cols_with_nan(puzzle)
    blocks_with_nan = find_blocks_with_nan(puzzle)

    # Each unknown cell gets enough qubits
    #unknown_regs = [QuantumRegister(nqubits_per_entry,
    #                                name="unknown{}".format(i))
    #                for i in range(num_unknown)]

    # Each known cell gets enough qubits
    #known_regs = [AncillaRegister(nqubits_per_entry,
    #                                name="known{}".format(i))
    #                for i in range(num_known)]

    # Rule registers
    # Flip if a sudoku rule is satisfied
    row_regs = [AncillaRegister(1, name="row{}".format(row))
                for row in rows_with_nan]
    col_regs = [AncillaRegister(1, name="col{}".format(col))
                for col in cols_with_nan]
    block_regs = [AncillaRegister(1, name="block{}".format(block))
                    for block in blocks_with_nan]

    # Oracle qubit, which should flip if all rules are satisfied
    oracle_qubit = AncillaRegister(1, name="q")

    # In a row, col, or block, there are `nrows` entries.
    # We need to check at most (nrows choose 2) pairs of entries and store
    # the results in (nrows choose 2) ancilla.
    ancilla = AncillaRegister(math.comb(nrows, 2), name="a")

    # We also need to `nqubits_per_entry` qubits for `is_equal` to work.
    work_qr = AncillaRegister(nqubits_per_entry, name="w")

    qc = QuantumCircuit(*unknown_dict.values(),
                        *known_dict.values(),
                        *row_regs,
                        *col_regs,
                        *block_regs,
                        oracle_qubit,
                        ancilla,
                        work_qr)

    # Prepare state of unknown_regs with hadamard transform
    qc.h(unknown_dict.values())

    # Prepare state of known_regs
    qc.compose(prepare_known_ancilla(puzzle).to_gate(),
                known_dict.values(),
                inplace=True)

    for i in range(niter):
        qc.compose(grover_iteration(puzzle).to_gate(),
                    qubits = [*unknown_dict.values(),
                                *known_dict.values(),
                                *col_regs,
                                *block_regs,
                                oracle_qubit,
                                ancilla,
                                work_qr],
                    inplace=True)

    return qc

def grover_iteration(puzzle: np.ndarray) -> QuantumCircuit:
    """Returns the Grover iteration circuit for the given puzzle.

    This function assumes the input is an n^2 by n^2 sudoku puzzle, with
    entries in the range [0, n^2-1], with a unique solution.
    The output is the corresponding quantum circuit implementing the
    Grover iteration, consisting of the oracle and diffuser.

    #TODO Rewrite to deal with zero or multiple solutions

    Parameters
    ----------
    puzzle : np.ndarray
        n^2 by n^2 array, representing the unsolved puzzle,
        where an empty cell is np.nan

    Returns
    -------
    QuantumCircuit
        Implements the Grover iteration for the given sudoku puzzle

    Examples
    --------
    #TODO
    """
    # Important constants
    nrows = puzzle.shape[0]
    num_unknown = find_num_unknown(puzzle)
    num_known = find_num_known(puzzle)
    nqubits_per_entry = find_nqubits_per_entry(puzzle)

    rows_with_nan = find_rows_with_nan(puzzle)
    cols_with_nan = find_cols_with_nan(puzzle)
    blocks_with_nan = find_blocks_with_nan(puzzle)

    # Each unknown cell gets enough qubits
    unknown_regs = [QuantumRegister(nqubits_per_entry,
                                    name="unknown{}".format(i))
                    for i in range(num_unknown)]

    # Each known cell gets enough qubits
    known_regs = [AncillaRegister(nqubits_per_entry,
                                    name="known{}".format(i))
                    for i in range(num_known)]

    # Rule registers
    # Flip if a sudoku rule is satisfied
    row_regs = [AncillaRegister(1, name="row{}".format(row))
                for row in rows_with_nan]
    col_regs = [AncillaRegister(1, name="col{}".format(col))
                for col in cols_with_nan]
    block_regs = [AncillaRegister(1, name="block{}".format(block))
                    for block in blocks_with_nan]

    # Oracle qubit, which should flip if all rules are satisfied
    oracle_qubit = AncillaRegister(1, name="q")

    # In a row, col, or block, there are `nrows` entries.
    # We need to check at most (nrows choose 2) pairs of entries and store
    # the results in (nrows choose 2) ancilla.
    ancilla = AncillaRegister(math.comb(nrows, 2), name="a")

    # We also need to `nqubits_per_entry` qubits for `is_equal` to work.
    work_qr = AncillaRegister(nqubits_per_entry, name="w")

    qc = QuantumCircuit(*unknown_regs,
                        *known_regs,
                        *row_regs,
                        *col_regs,
                        *block_regs,
                        oracle_qubit,
                        ancilla,
                        work_qr)

    qc.compose(oracle(puzzle).to_gate(),
                qubits=[*unknown_regs,
                        *known_regs,
                        *row_regs,
                        *col_regs,
                        *block_regs,
                        oracle_qubit,
                        ancilla,
                        work_qr],
                inplace=True)

    qc.compose(grover_diffuser(num_unknown * nqubits_per_entry).to_gate(),
                qubits=[*unknown_regs],
                inplace=True)

    return qc

def grover_diffuser(nqubits: int) -> QuantumCircuit:
    """Returns the Grover diffuser circuit on `num_qubits` qubits.
    """
    qr = QuantumRegister(nqubits, name="x")
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

    return qc

########################################################################
# The marker oracle
########################################################################

def oracle(puzzle: np.ndarray) -> QuantumCircuit:
    """Returns the marker oracle for the given puzzle.

    This function assumes the input is an n^2 by n^2 sudoku puzzle, with
    entries in the range [0, n^2-1], with a unique solution.
    The output is the corresponding quantum circuit implementing the
    marker oracle, which flips the oracle qubit if and only if it sees a
    valid solution to the puzzle.

    #TODO Rewrite to deal with zero or multiple solutions

    Parameters
    ----------
    puzzle : np.ndarray
        n^2 by n^2 array, representing the unsolved puzzle,
        where an empty cell is np.nan

    Returns
    -------
    QuantumCircuit
        Implements the marker oracle for the given sudoku puzzle

    Examples
    --------
    #TODO
    """
    # Important constants
    nrows = puzzle.shape[0]
    nqubits_per_entry = find_nqubits_per_entry(puzzle)

    rows_with_nan = find_rows_with_nan(puzzle)
    cols_with_nan = find_cols_with_nan(puzzle)
    blocks_with_nan = find_blocks_with_nan(puzzle)

    unknown_coords = tuple([tuple(x) for x in np.argwhere(np.isnan(puzzle))])
    known_coords = tuple([tuple(x) for x in np.argwhere(~np.isnan(puzzle))])

    unknown_dict = {unknown_coords[i]: i for i in range(len(unknown_coords))}
    known_dict = {known_coords[i]: i for i in range(len(known_coords))}

    # Each unknown cell gets enough qubits
    unknown_regs = [QuantumRegister(nqubits_per_entry,
                                    name="unknown{}".format(tup))
                    for tup in unknown_coords]

    # Each known cell gets enough qubits
    known_regs = [AncillaRegister(nqubits_per_entry,
                                    name="known{}".format(tup))
                    for tup in known_coords]

    # Rule registers
    # Flip if a sudoku rule is satisfied
    row_regs = [AncillaRegister(1, name="row{}".format(row))
                for row in rows_with_nan]
    col_regs = [AncillaRegister(1, name="col{}".format(col))
                for col in cols_with_nan]
    block_regs = [AncillaRegister(1, name="block{}".format(block))
                    for block in blocks_with_nan]

    # Oracle qubit, which should flip if all rules are satisfied
    oracle_qubit = AncillaRegister(1, name="q")

    # In a row, col, or block, there are `nrows` entries.
    # We need to check at most (nrows choose 2) pairs of entries and store
    # the results in (nrows choose 2) ancilla.
    ancilla = AncillaRegister(math.comb(nrows, 2), name="a")

    # We also need to `nqubits_per_entry` qubits for `is_equal` to work.
    work_qr = AncillaRegister(nqubits_per_entry, name="w")

    qc = QuantumCircuit(*unknown_regs,
                        *known_regs,
                        *row_regs,
                        *col_regs,
                        *block_regs,
                        oracle_qubit,
                        ancilla,
                        work_qr)

    # Row registers

    # Keep track of how many ancilla registers we are using
    index = 0

    # Keep track of row register index
    row_ireg = 0

    # Look at a row index from rows_with_nan
    for i in range(len(rows_with_nan)):

        # Look at all pairs of entries in this row
        for tup in combinations(range(nrows), 2):

            # rows_with_nan tells you irow
            # tup tells you icol
            entry1 = (rows_with_nan[i], tup[0])
            entry2 = (rows_with_nan[i], tup[1])

            # Find corresponding registers of pair

            if entry1 in unknown_coords:
                entry1_ireg = unknown_dict.get(entry1)
                if entry2 in unknown_coords:
                    entry2_ireg = unknown_dict.get(entry2)

                    # Flip ancilla[index] iff pair is not equal
                    qc.compose(is_equal(nqubits_per_entry),
                                qubits=[*(unknown_regs[entry1_ireg]),
                                        *(unknown_regs[entry2_ireg]),
                                        ancilla[index],
                                        *work_qr],
                                inplace=True)

                    qc.x(ancilla[index])

                elif entry2 in known_coords:
                    entry2_ireg = known_dict.get(entry2)

                    # Flip ancilla[index] iff pair is not equal
                    qc.compose(is_equal(nqubits_per_entry),
                                qubits=[*(unknown_regs[entry1_ireg]),
                                        *(known_regs[entry2_ireg]),
                                        ancilla[index],
                                        *work_qr],
                                inplace=True)

                    qc.x(ancilla[index])

            elif entry1 in known_coords:
                entry1_ireg = known_dict.get(entry1)
                if entry2 in unknown_coords:
                    entry2_ireg = unknown_dict.get(entry2)

                    # Flip ancilla[index] iff pair is not equal
                    qc.compose(is_equal(nqubits_per_entry),
                                qubits=[*(known_regs[entry1_ireg]),
                                        *(unknown_regs[entry2_ireg]),
                                        ancilla[index],
                                        *work_qr],
                                inplace=True)

                    qc.x(ancilla[index])

                elif entry2 in known_coords:
                    entry2_ireg = known_dict.get(entry2)

                    # Flip ancilla[index] iff pair is not equal
                    qc.compose(is_equal(nqubits_per_entry),
                                qubits=[*(known_regs[entry1_ireg]),
                                        *(unknown_regs[entry2_ireg]),
                                        ancilla[index],
                                        *work_qr],
                                inplace=True)

                    qc.x(ancilla[index])

            # Increment index
            index += 1

        # At this point, we have updated all ancilla for irow.
        # Flip row_regs[row_ireg] iff all ancilla are 1
        qc.mcx(ancilla[:index], row_regs[row_ireg])

        # Uncompute ancilla
        # HOW?

    # Column registers

    # Block registers

    # Uncompute ancilla
    pass
