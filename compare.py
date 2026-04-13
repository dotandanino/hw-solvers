# Put your code here 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from time import perf_counter   # Exact measurement of CPU time


def measure_time(func, *args, **kwargs):
    """ return the run-time of the given function, in fractional seconds. """
    start = perf_counter()
    result = func(*args, **kwargs)
    end = perf_counter()
    return end-start


def solveWithRoot(a: np.ndarray,b: np.ndarray):
    """
    Solves a system of linear equations Ax = b using scipy.optimize.root.

    The problem is reformulated as finding the root of f(x) = Ax - b.

    Parameters:
    a (array_like): A 2D array representing the coefficient matrix A.
    b (array_like): A 1D array representing the ordinate values b.

    Returns:
    ndarray: A 1D array containing the solution vector x.

    Examples:
    # defualt case
    >>> import numpy as np
    >>> a = np.array([[3, 1], [1, 2]])
    >>> b = np.array([9, 8])
    >>> x = solveWithRoot(a, b)
    >>> # Using np.allclose to avoid floating-point precision issues in testing
    >>> np.allclose(x, [2., 3.])
    True

    # case from the lecture
    >>> import numpy as np
    >>> a = np.array([[1,  1],[1.5,4]])
    >>> b = np.array([2200,5050])
    >>> x = solveWithRoot(a, b)
    >>> np.allclose(x, [1500., 700.])
    True

    # case with 3 args
    >>> a3 = np.array([[1, 1, 1], [1, -1, 2], [2, 3, -1]])
    >>> b3 = np.array([6, 5, 5])
    >>> x3 = solveWithRoot(a3, b3)
    >>> np.allclose(x3, [1., 2., 3.]) 
    True

    #case with 1 arg
    >>> a1 = np.array([[5]])
    >>> b1 = np.array([10])
    >>> x1 = solveWithRoot(a1, b1)
    >>> np.allclose(x1, [2.])
    True

    #case with negative and flaot
    >>> a_float = np.array([[0.5, -0.5], [0.2, 0.8]])
    >>> b_float = np.array([1.0, 2.4])
    >>> x_float = solveWithRoot(a_float, b_float)
    >>> np.allclose(x_float, [4., 2.])
    True

    """

    start = np.zeros(len(b))
    # ax-b = 0 instead of ax = b
    sol = root(lambda x: a @ x - b, x0=start)
    return sol.x

def test_a(num_tests=100, max_dim=10):
    """
    test the function I wrote before using umpy.linalg.solve
    Parameters:
    num_tests - amount of test
    max_dim - maximum amount of args

    """
    tests_passed = 0
    
    for i in range(num_tests):
        n = np.random.randint(1, max_dim + 1)
        a = np.random.randn(n, n)
        b = np.random.randn(n)
        expected_solution = np.linalg.solve(a, b)
        our_solution = solveWithRoot(a, b)
        if np.allclose(expected_solution, our_solution):
            tests_passed += 1
        else:
            print(f"Test failed for size {n}x{n}")
                
    print(f"Successfully passed {tests_passed}/{num_tests} random tests!")

def compare_times(comparision_num = 51):
    sizes = np.linspace(1,1000,comparision_num,dtype = int)

    times_root = []
    times_linalg = []
    
    print("start measuring time")
    
    for n in sizes:
        a = np.random.randn(n, n) + np.eye(n) * n
        b = np.random.randn(n)
        
        time_linalg = measure_time(np.linalg.solve, a, b)
        times_linalg.append(time_linalg)
        
        time_root = measure_time(solveWithRoot, a, b)
        times_root.append(time_root)
        

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_root, label='scipy.optimize.root', color='red')
    plt.plot(sizes, times_linalg, label='numpy.linalg.solve', color='blue')
    
    plt.xlabel('Matrix Size (n)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Performance Comparison: Root Finding vs Linear Solver')
    plt.legend()
    plt.grid(True)
    
    #plt.show()
    plt.savefig("comparison.png")

if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
    test_a(100,10)
    compare_times()