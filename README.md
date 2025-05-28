# Benchmark Functions Library

A collection of benchmark mathematical functions for optimization purposes. This library provides a set of well-known mathematical functions commonly used in optimization algorithms, machine learning, and mathematical research.

## Features

- Collection of 8 benchmark functions:
  - Ackley function
  - Gramacy and Lee function
  - Griewank function
  - Rastrigin function
  - Rosenbrock function
  - Schubert function
  - Schwefel function
  - Forrester function

## Installation

```bash
pip install benchmark_functions-0.1.0-py3-none-any.whl
```

## Quick Start

```python
from benchmark_functions import Ackley, Rastrigin, Rosenbrock

# Create function instances
ackley = Ackley()
rastrigin = Rastrigin()
rosenbrock = Rosenbrock()

# Evaluate functions
x = [0.5, 0.5]  # 2D point
result = ackley(x)
print(f"Ackley function value at {x}: {result}")

# Get function bounds
bounds = ackley.bounds
print(f"Function bounds: {bounds}")
```

## Features

- Easy-to-use interface
- Vectorized operations using NumPy
- Proper bounds and domain information
- Visualization capabilities
- Comprehensive documentation
- Unit tests

## Requirements

- Python >= 3.8
- NumPy >= 1.21.0
- SciPy >= 1.7.0

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite it as:

```bibtex
@software{benchmark_functions,
  author = {Benamis},
  title = {Benchmark Functions Library},
  year = {2025},
}
``` 
