"""
Qiskit Quantum Adder for adding two 3-bit numbers (0-7)
Uses CNOT and Toffoli gates to perform actual quantum addition
Compatible with Qiskit 2.x
"""

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
import qiskit_aer

# Display version information
print(f"Qiskit version: {qiskit.__version__}")
print(f"Qiskit Aer version: {qiskit_aer.__version__}")
print()


def quantum_adder_with_cnot(num1, num2):
    """
    Quantum adder using CNOT gates for two 3-bit numbers (0-7)
    
    This implements a proper quantum ripple-carry adder using:
    - CNOT gates for XOR operations (sum bits)
    - Toffoli (CCX) gates for AND operations (carry bits)
    
    Args:
        num1: First number (0-7)
        num2: Second number (0-7)
    
    Returns:
        QuantumCircuit: The quantum circuit for addition
    """
    # Validate inputs
    if not (0 <= num1 <= 7 and 0 <= num2 <= 7):
        raise ValueError("Numbers must be between 0 and 7 (inclusive)")
    
    # Convert numbers to binary (3 bits each)
    bin1 = format(num1, '03b')
    bin2 = format(num2, '03b')
    
    print(f"Adding {num1} (binary: {bin1}) + {num2} (binary: {bin2})")
    
    n = 3  # number of bits
    
    # Create quantum registers
    # a: first number (3 qubits)
    # b: second number (3 qubits) - will hold the sum
    # c: carry qubits (4 qubits: c[0] is always 0, c[1-3] for carries, c[3] is final carry out)
    a = QuantumRegister(n, 'a')
    b = QuantumRegister(n, 'b')
    c = QuantumRegister(n + 1, 'c')
    
    # Classical register for result (4 bits: 3 sum bits + 1 carry out)
    result = ClassicalRegister(n + 1, 'result')
    
    qc = QuantumCircuit(a, b, c, result)
    
    # Initialize inputs (LSB is index 0)
    for i in range(n):
        if bin1[n-1-i] == '1':
            qc.x(a[i])
        if bin2[n-1-i] == '1':
            qc.x(b[i])
    
    qc.barrier(label='Input')
    
    # Ripple-carry adder using CNOT and Toffoli gates
    # For each bit position i:
    # 1. Compute carry: c[i+1] = (a[i] AND b[i]) OR (c[i] AND (a[i] XOR b[i]))
    # 2. Compute sum: b[i] = a[i] XOR b[i] XOR c[i]
    
    # Bit 0 (LSB) - no carry in
    qc.ccx(a[0], b[0], c[1])  # c[1] = a[0] AND b[0]
    qc.cx(a[0], b[0])          # b[0] = a[0] XOR b[0] (sum bit 0)
    
    qc.barrier(label='Bit 0')
    
    # Bit 1
    # Carry: c[2] = (a[1] AND b[1]) OR (c[1] AND (a[1] XOR b[1]))
    qc.ccx(a[1], b[1], c[2])   # Part 1: a[1] AND b[1]
    qc.cx(a[1], b[1])           # b[1] = a[1] XOR b[1]
    qc.ccx(c[1], b[1], c[2])   # Part 2: c[1] AND (a[1] XOR b[1])
    qc.cx(c[1], b[1])           # b[1] = a[1] XOR b[1] XOR c[1] (sum bit 1)
    
    qc.barrier(label='Bit 1')
    
    # Bit 2 (MSB of input)
    # Carry: c[3] = (a[2] AND b[2]) OR (c[2] AND (a[2] XOR b[2]))
    qc.ccx(a[2], b[2], c[3])   # Part 1: a[2] AND b[2]
    qc.cx(a[2], b[2])           # b[2] = a[2] XOR b[2]
    qc.ccx(c[2], b[2], c[3])   # Part 2: c[2] AND (a[2] XOR b[2])
    qc.cx(c[2], b[2])           # b[2] = a[2] XOR b[2] XOR c[2] (sum bit 2)
    
    qc.barrier(label='Bit 2')
    
    # Measure results
    # b register now contains the sum bits (LSB to MSB)
    # c[3] contains the final carry out
    for i in range(n):
        qc.measure(b[i], result[i])
    qc.measure(c[3], result[3])
    
    return qc


def run_quantum_adder(num1, num2, shots=1024, show_circuit=True):
    """
    Run the quantum adder and display results
    
    Args:
        num1: First number (0-7)
        num2: Second number (0-7)
        shots: Number of times to run the circuit
        show_circuit: Whether to display the circuit diagram
    
    Returns:
        int: The result of the addition
    """
    # Create the quantum circuit
    qc = quantum_adder_with_cnot(num1, num2)
    
    # Display circuit if requested
    if show_circuit:
        print("\nQuantum Circuit:")
        try:
            circuit_str = qc.draw(output='text')
            print(circuit_str)
        except (UnicodeEncodeError, UnicodeDecodeError):
            print(f"Circuit with {qc.num_qubits} qubits and {qc.depth()} depth")
            print(f"Gates used: {qc.count_ops()}")
            print("Note: Circuit uses CNOT (CX) and Toffoli (CCX) gates for addition")
    
    # Get the Aer simulator
    simulator = Aer.get_backend('qasm_simulator')
    
    # Transpile and run the circuit
    transpiled_qc = transpile(qc, simulator)
    job = simulator.run(transpiled_qc, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    # Get the most common result
    most_common = max(counts, key=counts.get)
    
    # Qiskit returns measurement results with highest index on the left
    # String format: c[3] c[2] c[1] c[0] which is already MSB to LSB
    result_decimal = int(most_common, 2)
    
    print(f"\nMeasurement: {most_common} (binary)")
    print(f"Result: {num1} + {num2} = {result_decimal}")
    print(f"Expected: {num1 + num2}")
    print(f"Status: {'PASS' if result_decimal == num1 + num2 else 'FAIL'}")
    
    # Show gate statistics
    gate_counts = qc.count_ops()
    print(f"\nGate Statistics:")
    print(f"  CNOT gates (CX): {gate_counts.get('cx', 0)}")
    print(f"  Toffoli gates (CCX): {gate_counts.get('ccx', 0)}")
    print(f"  Total quantum gates: {gate_counts.get('cx', 0) + gate_counts.get('ccx', 0)}")
    
    return result_decimal


def test_all_combinations():
    """
    Test the quantum adder with all 64 possible combinations (8x8)
    Tests every pair from (0,0) to (7,7)
    """
    print("=" * 70)
    print("Testing Quantum Adder with CNOT Gates - All 64 Combinations")
    print("=" * 70)
    
    passed = 0
    failed = 0
    failed_cases = []
    
    # Test all 8x8 = 64 combinations
    for num1 in range(8):
        for num2 in range(8):
            result = run_quantum_adder(num1, num2, shots=100, show_circuit=False)
            expected = num1 + num2
            
            if result == expected:
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"
                failed_cases.append((num1, num2, result, expected))
            
            # Print compact result
            print(f"{num1} + {num2} = {result} (expected {expected}) [{status}]")
    
    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed} passed, {failed} failed out of 64 tests")
    
    if failed > 0:
        print(f"\nFailed cases:")
        for num1, num2, result, expected in failed_cases:
            print(f"  {num1} + {num2} = {result} (expected {expected})")
    
    if failed == 0:
        print("\nSUCCESS: All 64 test cases passed!")
    else:
        print(f"\nFAILURE: {failed} test(s) failed")
    print("=" * 70)
    
    return passed, failed


def demo():
    """Run a demonstration of the quantum adder"""
    print("=" * 70)
    print("Quantum Adder Using CNOT and Toffoli Gates")
    print("=" * 70)
    print("\nThis quantum adder uses:")
    print("- CNOT (CX) gates for XOR operations (computing sum bits)")
    print("- Toffoli (CCX) gates for AND operations (computing carry bits)")
    print("- Implements a ripple-carry adder algorithm")
    print("\nIt can add two numbers between 0 and 7.\n")
    
    # Example additions
    examples = [(3, 5), (7, 7), (4, 2)]
    
    for num1, num2 in examples:
        print("\n" + "-" * 70)
        run_quantum_adder(num1, num2, shots=1024, show_circuit=True)
        print("-" * 70)


if __name__ == "__main__":
    # Run demonstration
    demo()
    
    print("\n\n")
    
    # Run comprehensive tests
    test_all_combinations()