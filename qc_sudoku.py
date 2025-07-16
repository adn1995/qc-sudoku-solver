# filename: qc_sudoku.py
# author: Arthur Diep-Nguyen

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.quantum_info import Statevector

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
        arr_nums = np.nan_to_num(arr) # Turn nans to zeros
        return np.all((arr_nums >= 0) | (arr_nums < nrows))

def is_sq_matrix(arr: np.ndarray) -> bool:
    """Returns if an array is a square matrix.
    """
    size = arr.size
    nrows = math.isqrt(size)
    return (arr.shape == (nrows,nrows))

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

def find_rows_with_nan(puzzle: np.ndarray) -> tuple[int, ...]:
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

def find_cols_with_nan(puzzle: np.ndarray) -> tuple[int, ...]:
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

def find_blocks_with_nan(puzzle: np.ndarray) -> tuple[int, ...]:
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

def make_unknown_dict(puzzle: np.ndarray) -> dict[tuple[int, int], QuantumRegister]:
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
                                    name="unknown_({},{})".format(i,j))
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
                                    name="known_({},{})".format(i,j))
            for i in range(nrows) for j in range(nrows)
            if not np.isnan(puzzle)[i,j]}

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
    """
    output_state = Statevector(qc)
    prob_dict = output_state.probabilities_dict(qargs)
    return max(prob_dict, key=prob_dict.get)

def value_to_ancilla(value: int, nqubits: int) -> QuantumCircuit:
    """Returns a quantum circuit that stores the given value in the
    ancilla register.

    Parameters
    ----------
    value : int
        Value to be stored
    nqubits : int
        Size of the register where the value will be stored
    """
    qr = AncillaRegister(nqubits, name="a")
    qc = QuantumCircuit(qr, name="Set value")

    # If value == 0, we want an empty gate
    if value == 0:
        return qc
    else:
        bitstr = int_to_bitstr(value, nqubits)
        assert len(bitstr) == nqubits

        # Note that higher index in the register means more significant
        # so we need to reverse the bitstring
        bitstr = bitstr[::-1]

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
        if not math.isnan(x):
            qc.compose(value_to_ancilla(int(x), nqubits_per_entry),
                        qr[index:index+nqubits_per_entry],
                        inplace=True)
            index += nqubits_per_entry
    assert index == nqubits

    return qc

def is_equal(nqubits: int) -> QuantumCircuit:
    """Returns a quantum circuit that flips an output qubit if two input
    registers have the same state.

    Parameters
    ----------
    nqubits : int
        Size of each of the two input registers

    Returns
    -------
    QuantumCircuit
    """
    in_qr1 = QuantumRegister(nqubits, name="x")
    in_qr2 = QuantumRegister(nqubits, name="y")
    out_qr = AncillaRegister(1, name="out")

    # We need to do `nqubits` comparisons, so we need that many qubits
    work_qr = AncillaRegister(nqubits, name="w")

    qc = QuantumCircuit(in_qr1, in_qr2, out_qr, work_qr, name="Is equal")

    # Store comparisons in work ancilla
    for i in range(nqubits):
        qc.compose(XOR_qc().to_gate(),
                    qubits=[in_qr1[i], in_qr2[i], work_qr[i]],
                    inplace=True)

    # Flip target_qr iff all work ancilla are 1
    # We are allowed to use multi-controlled X because this function
    # is used only in the marker oracle
    qc.mcx(work_qr, out_qr)

    # Uncompute ancilla
    for i in range(nqubits):
        qc.compose(XOR_qc().to_gate(),
                    qubits=[in_qr1[i], in_qr2[i], work_qr[i]],
                    inplace=True)

    return qc

def is_valid_group(group: np.ndarray,
                    nqubits_per_entry: int,
                    unknown_regs: tuple[QuantumRegister, ...],
                    known_qr: AncillaRegister,
                    group_qr: AncillaRegister,
                    check_qr: AncillaRegister,
                    work_qr: AncillaRegister
    ) -> QuantumCircuit:
    """Returns a quantum circuit that checks if the grouping is valid.

    Let's call a row, column, or block, a "group" of cells.
    Given a group, this function flips an output register iff all
    cells in the grouping are distinct.

    Parameters
    ----------
    group : np.ndarray
        Row, column, or block of the puzzle
    nqubits_per_entry : int
        Number of qubits needed to represent a single cell of the puzzle
    unknown_regs : tuple[QuantumRegister, ...]
        Each QuantumRegister corresponds to an unknown cell in `group`
    known_qr : AncillaRegister
        Used for preparing state of filled cell
    group_qr : AncillaRegister
        Stores rule violations for the given group
    check_qr : AncillaRegister
        Used for storing results of equality checks
    work_qr : AncillaRegister
        Used for computing equality checks

    Returns
    -------
    QuantumCircuit
    """
    ncells = len(group)

    qc = QuantumCircuit(*unknown_regs,
                        known_qr,
                        group_qr,
                        check_qr,
                        work_qr,
                        name="Is valid group")

    check_qc = QuantumCircuit(*unknown_regs,
                        known_qr,
                        group_qr,
                        check_qr,
                        work_qr,
                        name="Check equalities")

    # Use check_qr to store results of equality checks
    check_index = 0

    # Map indices to unknown_regs
    assert np.count_nonzero(np.isnan(group)) == len(unknown_regs)
    index_to_reg = dict()
    unknown_count = 0
    for i in range(len(group)):
        if math.isnan(group[i]):
            index_to_reg.update({i: unknown_regs[unknown_count]})
            unknown_count += 1
    assert unknown_count == len(unknown_regs)
    assert len(index_to_reg) == len(unknown_regs)

    for (i,j) in combinations(range(ncells),2):

        # If both cells are empty, compare them to each other
        if math.isnan(group[i]) and math.isnan(group[j]):
            check_qc.compose(is_equal(nqubits_per_entry).to_gate(),
                                qubits=[*index_to_reg.get(i),
                                        *index_to_reg.get(j),
                                        check_qr[check_index],
                                        *work_qr],
                                inplace=True)

        # If first cell is empty and second cell is known,
        # then prepare known_qr with the second cell, then compare
        elif math.isnan(group[i]):

            # Prepare known_qr
            check_qc.compose(value_to_ancilla(int(group[j]),
                                                nqubits_per_entry
                                                ).to_gate(),
                                known_qr,
                                inplace=True)

            # Check equality
            check_qc.compose(is_equal(nqubits_per_entry).to_gate(),
                                qubits=[*index_to_reg.get(i),
                                        *known_qr,
                                        check_qr[check_index],
                                        *work_qr],
                                inplace=True)

            # Unprepare known_qr
            # Notice that value_to_ancilla is its own inverse
            check_qc.compose(value_to_ancilla(int(group[j]),
                                                nqubits_per_entry
                                                ).to_gate(),
                                known_qr,
                                inplace=True)
        # If second cell is empty and first cell is known,
        # then prepare known_qr with the first cell, then compare
        elif math.isnan(group[j]):
            # Prepare known_qr
            check_qc.compose(value_to_ancilla(int(group[i]),
                                                nqubits_per_entry
                                                ).to_gate(),
                                known_qr,
                                inplace=True)

            # Check equality
            check_qc.compose(is_equal(nqubits_per_entry).to_gate(),
                                qubits=[*index_to_reg.get(j),
                                        *known_qr,
                                        check_qr[check_index],
                                        *work_qr],
                                inplace=True)

            # Unprepare known_qr
            # Notice that value_to_ancilla is its own inverse
            check_qc.compose(value_to_ancilla(int(group[i]),
                                                nqubits_per_entry
                                                ).to_gate(),
                                known_qr,
                                inplace=True)
        # If both cells are known, then we don't need to do anything
        check_index += 1

    assert check_index == math.comb(ncells,2)
    qc.compose(check_qc, inplace=True)

    # Flip group_qr iff all qubits in check_qr are flipped
    qc.mcx(control_qubits=check_qr,
            target_qubit=group_qr)

    # Uncompute check_qr ancilla
    # Notice that is_equal and value_to_ancilla are their own inverses
    # so we need only compose check_qc
    # but let's compose check_qc.inverse() just in case
    qc.compose(check_qc.inverse(), inplace=True)

    return qc

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
    """
    # Important constants
    nrows = puzzle.shape[0]
    nqubits_per_entry = find_nqubits_per_entry(puzzle)

    # Dictionary with quantum registers for each unknown cell
    unknown_dict = make_unknown_dict(puzzle)

    # Register where state of known cells are prepared
    known_qr = AncillaRegister(nqubits_per_entry, name="known")

    rows_with_nan = find_rows_with_nan(puzzle)
    cols_with_nan = find_cols_with_nan(puzzle)
    blocks_with_nan = find_blocks_with_nan(puzzle)

    # Rule registers
    # Flip if a sudoku rule is satisfied
    row_regs = {row : AncillaRegister(1, name="row{}".format(row))
                for row in rows_with_nan}
    col_regs = {col : AncillaRegister(1, name="col{}".format(col))
                for col in cols_with_nan}
    block_regs = {block : AncillaRegister(1, name="block{}".format(block))
                    for block in blocks_with_nan}

    # Oracle qubit, which should flip if all rules are satisfied
    oracle_qr = AncillaRegister(1, name="oracle")

    # In a row, col, or block, there are `nrows` entries.
    # We need to check at most (nrows choose 2) pairs of entries and store
    # the results in (nrows choose 2) ancilla.
    check_qr = AncillaRegister(math.comb(nrows, 2), name="check")

    # We also need to `nqubits_per_entry` qubits for `is_equal` to work.
    work_qr = AncillaRegister(nqubits_per_entry, name="work")

    qc = QuantumCircuit(*unknown_dict.values(),
                        known_qr,
                        *row_regs.values(),
                        *col_regs.values(),
                        *block_regs.values(),
                        oracle_qr,
                        check_qr,
                        work_qr,
                        name="Grover circuit")

    # Prepare state of unknown_regs with Hadamard transform
    for reg in unknown_dict.values():
        qc.h(reg)

    # Prepare oracle qubit into the "minus" state
    #   (|0> - |1>)/sqrt(2)
    qc.h(oracle_qr)
    qc.z(oracle_qr)

    # Prepare check_qr to the all-ones state
    # When the oracle checks a grouping, it first assumes that no rules
    # are violated, i.e. check_qr is in the all-ones state.
    # A rule violation will cause a qubit of check_qr to flip to zero.
    # If at least one of these qubits is zero, then the MCX gate
    # controlled by check_qr will not flip the register for the grouping.
    qc.x(check_qr)

    # Do Grover iterations
    for i in range(niter):
        qc.compose(grover_iteration(puzzle=puzzle,
                                    nrows=nrows,
                                    nqubits_per_entry=nqubits_per_entry,
                                    unknown_dict=unknown_dict,
                                    known_qr=known_qr,
                                    row_regs=row_regs,
                                    col_regs=col_regs,
                                    block_regs=block_regs,
                                    oracle_qr=oracle_qr,
                                    check_qr=check_qr,
                                    work_qr=work_qr).to_gate(),
                    inplace=True)

    return qc

def grover_iteration(puzzle: np.ndarray,
                    nrows: int,
                    nqubits_per_entry: int,
                    unknown_dict: dict[tuple[int, int], QuantumRegister],
                    known_qr: AncillaRegister,
                    row_regs: dict[int, AncillaRegister],
                    col_regs: dict[int, AncillaRegister],
                    block_regs: dict[int, AncillaRegister],
                    oracle_qr: AncillaRegister,
                    check_qr: AncillaRegister,
                    work_qr: AncillaRegister
    ) -> QuantumCircuit:
    """Returns the Grover iteration circuit for the given parameters.

    #TODO Rewrite to deal with zero or multiple solutions

    Parameters
    ----------
    puzzle : np.ndarray
        n^2 by n^2 array, representing the unsolved puzzle,
        where an empty cell is np.nan
    nrows : int
        Number of rows in the puzzle
    nqubits_per_entry : int
        Number of qubits needed to represent a single cell of the puzzle
    unknown_dict : dict[tuple[int, int], QuantumRegister]
        Maps coordinates of an empty cell to a QuantumRegister
    known_qr : AncillaRegister
        Used for preparing state of filled cell
    row_regs : dict[int, AncillaRegister]
        Maps row index to a register for storing rule violations
    col_regs : dict[int, AncillaRegister]
        Maps col index to a register for storing rule violations
    block_regs : dict[int, AncillaRegister]
        Maps block index to a register for storing rule violations
    oracle_qr : AncillaRegister
        Oracle qubit, which flips when all sudoku rules are satisfied
    check_qr : AncillaRegister
        Used for storing results of equality checks
    work_qr : AncillaRegister
        Used for computing equality checks

    Returns
    -------
    QuantumCircuit
        Implements the Grover iteration for the given sudoku puzzle
    """

    qc = QuantumCircuit(*unknown_dict.values(),
                        known_qr,
                        *row_regs.values(),
                        *col_regs.values(),
                        *block_regs.values(),
                        oracle_qr,
                        check_qr,
                        work_qr,
                        name="Grover iteration")

    qc.compose(oracle(puzzle=puzzle,
                        nrows=nrows,
                        nqubits_per_entry=nqubits_per_entry,
                        unknown_dict=unknown_dict,
                        known_qr=known_qr,
                        row_regs=row_regs,
                        col_regs=col_regs,
                        block_regs=block_regs,
                        oracle_qr=oracle_qr,
                        check_qr=check_qr,
                        work_qr=work_qr).to_gate(),
                inplace=True)

    unknown_qubits = []
    for reg in unknown_dict.values():
        unknown_qubits.extend([*reg])
    qc.compose(grover_diffuser(len(unknown_qubits)).to_gate(),
                unknown_qubits,
                inplace=True)

    return qc

def grover_diffuser(nqubits: int) -> QuantumCircuit:
    """Returns the Grover diffuser circuit on `nqubits` qubits.

    This circuit is used to perform a conditional phase shift on the
    registers corresponding to unknown cells of the puzzle.
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

def oracle(puzzle: np.ndarray,
            nrows: int,
            nqubits_per_entry: int,
            unknown_dict: dict[tuple[int, int], QuantumRegister],
            known_qr: AncillaRegister,
            row_regs: dict[int, AncillaRegister],
            col_regs: dict[int, AncillaRegister],
            block_regs: dict[int, AncillaRegister],
            oracle_qr: AncillaRegister,
            check_qr: AncillaRegister,
            work_qr: AncillaRegister) -> QuantumCircuit:
    """Returns the marker oracle for the given parameters.

    This function assumes that the parameters correspond to an n^2 by
    n^2 sudoku puzzle, with entries in the range [0, n^2-1], with a
    unique solution.
    The output is the corresponding quantum circuit implementing the
    marker oracle, which flips the oracle qubit if and only if it sees a
    valid solution to the puzzle.

    #TODO Rewrite to deal with zero or multiple solutions

    Parameters
    ----------
    puzzle : np.ndarray
        n^2 by n^2 array, representing the unsolved puzzle,
        where an empty cell is np.nan
    nrows : int
        Number of rows in the puzzle
    nqubits_per_entry : int
        Number of qubits needed to represent a single cell of the puzzle
    unknown_dict : dict[tuple[int, int], QuantumRegister]
        Maps coordinates of an empty cell to a QuantumRegister
    known_qr : AncillaRegister
        Used for preparing state of filled cell
    row_regs : dict[int, AncillaRegister]
        Maps row index to a register for storing rule violations
    col_regs : dict[int, AncillaRegister]
        Maps col index to a register for storing rule violations
    block_regs : dict[int, AncillaRegister]
        Maps block index to a register for storing rule violations
    oracle_qr : AncillaRegister
        Oracle qubit, which flips when all sudoku rules are satisfied
    check_qr : AncillaRegister
        Used for storing results of equality checks
    work_qr : AncillaRegister
        Used for computing equality checks

    Returns
    -------
    QuantumCircuit
        Implements the marker oracle for the given sudoku puzzle
    """
    n = math.isqrt(nrows)

    qc = QuantumCircuit(*unknown_dict.values(),
                        known_qr,
                        *row_regs.values(),
                        *col_regs.values(),
                        *block_regs.values(),
                        oracle_qr,
                        check_qr,
                        work_qr,
                        name="Oracle")

    # Subcircuit for checking all groupings for rule violations
    # Compute the inverse later to uncompute row, col, and block regs
    check_qc = QuantumCircuit(*unknown_dict.values(),
                                known_qr,
                                *row_regs.values(),
                                *col_regs.values(),
                                *block_regs.values(),
                                oracle_qr,
                                check_qr,
                                work_qr,
                                name="Check rules")

    # Row registers
    for row in row_regs:

        unknown_regs = []
        unknown_qubits = []

        for col in range(nrows):
            if (row,col) in unknown_dict:
                reg = unknown_dict.get((row,col))
                unknown_regs.append(reg)
                unknown_qubits.extend([*reg])
        unknown_regs = tuple(unknown_regs)

        group_qr = row_regs.get(row)

        check_qc.compose(is_valid_group(group=puzzle[row,:],
                                        nqubits_per_entry=nqubits_per_entry,
                                        unknown_regs=unknown_regs,
                                        known_qr=known_qr,
                                        group_qr=group_qr,
                                        check_qr=check_qr,
                                        work_qr=work_qr),
                        qubits=[*unknown_qubits,
                                *known_qr,
                                *group_qr,
                                *check_qr,
                                *work_qr],
                        inplace=True)

    # Column registers
    for col in col_regs:

        unknown_regs = []
        unknown_qubits = []

        for row in range(nrows):
            if (row,col) in unknown_dict:
                reg = unknown_dict.get((row,col))
                unknown_regs.append(reg)
                unknown_qubits.extend([*reg])
        unknown_regs = tuple(unknown_regs)

        group_qr = col_regs.get(col)

        check_qc.compose(is_valid_group(group=puzzle[:,col],
                                        nqubits_per_entry=nqubits_per_entry,
                                        unknown_regs=unknown_regs,
                                        known_qr=known_qr,
                                        group_qr=group_qr,
                                        check_qr=check_qr,
                                        work_qr=work_qr),
                        qubits=[*unknown_qubits,
                                *known_qr,
                                *group_qr,
                                *check_qr,
                                *work_qr],
                        inplace=True)

    # Block registers
    for block in block_regs:

        unknown_regs = []
        unknown_qubits = []

        cell_coords = [(n*(block // n)+i, n*(block % n)+j)
                        for i
                        in range(n)
                        for j
                        in range(n)]

        for coord in cell_coords:
            if coord in unknown_dict:
                reg = unknown_dict.get(coord)
                unknown_regs.append(reg)
                unknown_qubits.extend([*reg])
        unknown_regs = tuple(unknown_regs)

        group_qr = block_regs.get(block)
        group = puzzle[n*(block // n) : n*(block // n) + n,
                        n*(block % n) : n*(block % n) + n]
        group = group.flatten()

        check_qc.compose(is_valid_group(group=group,
                                        nqubits_per_entry=nqubits_per_entry,
                                        unknown_regs=unknown_regs,
                                        known_qr=known_qr,
                                        group_qr=group_qr,
                                        check_qr=check_qr,
                                        work_qr=work_qr),
                        qubits=[*unknown_qubits,
                                *known_qr,
                                *group_qr,
                                *check_qr,
                                *work_qr],
                        inplace=True)

    # Compose checks to oracle circuit
    qc.compose(check_qc.to_gate(),
                inplace=True)

    # Multicontrolled X to oracle_qubit
    qc.mcx(control_qubits=[*row_regs.values(),
                            *col_regs.values(),
                            *block_regs.values()],
            target_qubit=oracle_qr)

    # Uncompute row_regs, col_regs, block_regs using check_qc.inverse()
    qc.compose(check_qc.inverse().to_gate(),
                inplace=True)

    return qc
