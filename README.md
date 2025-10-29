# Matrix Filler

Fill missing matrix values (np.nan) to satisfy row/column sum constraints using bounded least squares optimization.

## Overview

Matrix Filler is a Python package that solves the problem of filling unknown values in a 2D matrix when you know the desired row and column sums. It uses scipy's bounded least squares optimization to find the best values that satisfy the constraints.

## Features

- **Constraint-based filling**: Automatically fills missing values to match specified row and column sums
- **Optimization-based approach**: Uses bounded least squares for robust solutions
- **Flexible bounds**: Optional non-negative constraint for filled values
- **Detailed results**: Returns both the filled matrix and optimization diagnostics

## Installation

### From PyPI

```bash
pip install matrix-filler
```

## Requirements

- Python >= 3.10
- numpy
- scipy

## Usage

### Basic Example

```python
import numpy as np
from matrix_filler import fill_matrix_with_constraints

# Create a matrix with missing values (np.nan)
grid = np.array([
    [1.0, np.nan, 3.0],
    [np.nan, 2.0, np.nan],
    [4.0, np.nan, 5.0]
])

# Define desired row and column sums
row_targets = np.array([10.0, 15.0, 20.0])  # Sum for each row
col_targets = np.array([12.0, 8.0, 25.0])   # Sum for each column

# Fill the matrix
filled_grid, result = fill_matrix_with_constraints(
    grid, 
    row_targets, 
    col_targets, 
    non_negative=True
)

print("Filled Matrix:")
print(filled_grid)
print(f"\nOptimization Success: {result.success}")
print(f"Message: {result.message}")
```

### Advanced Example with Validation

```python
import numpy as np
from matrix_filler import fill_matrix_with_constraints

# Create a partially filled matrix
grid = np.array([
    [10.0, np.nan, np.nan],
    [np.nan, 20.0, np.nan],
    [np.nan, np.nan, 30.0]
])

row_targets = np.array([50.0, 60.0, 70.0])
col_targets = np.array([40.0, 80.0, 60.0])

# Fill the matrix (allow negative values)
filled_grid, result = fill_matrix_with_constraints(
    grid, 
    row_targets, 
    col_targets, 
    non_negative=False
)

# Verify the constraints
print("Row sums:", filled_grid.sum(axis=1))
print("Column sums:", filled_grid.sum(axis=0))
print("\nExpected row sums:", row_targets)
print("Expected column sums:", col_targets)
```

## API Reference

### `fill_matrix_with_constraints(grid, row_targets, col_targets, non_negative=True)`

Fill np.nan entries in a 2D numpy array so that row and column sums match given targets.

#### Parameters

- **grid** (`np.ndarray`): 2D array with known values and `np.nan` for unknowns
- **row_targets** (`np.ndarray`): Desired sum of each row (length = number of rows)
- **col_targets** (`np.ndarray`): Desired sum of each column (length = number of columns)
- **non_negative** (`bool`, optional): If `True`, forces all filled values to be >= 0. Default is `True`

#### Returns

- **filled_grid** (`np.ndarray`): Copy of grid with all `np.nan` values filled
- **result** (`scipy.optimize.OptimizeResult`): SciPy optimization result containing:
  - `x`: Array of solved unknowns
  - `success`: Boolean indicating if optimization succeeded
  - `message`: String describing the optimization outcome
  - Additional optimization metrics

## How It Works

1. **Identify unknowns**: Locates all `np.nan` positions in the input matrix
2. **Build linear system**: Constructs a system of linear equations `Ax = b` where:
   - Each row constraint contributes one equation
   - Each column constraint contributes one equation
   - Variables are the unknown matrix entries
3. **Apply bounds**: Sets lower bounds (0 or -∞) and upper bounds (+∞) for variables
4. **Solve optimization**: Uses `scipy.optimize.lsq_linear` to find the best least-squares solution
5. **Fill matrix**: Inserts the solved values back into the matrix

## Use Cases

- **Data imputation**: Fill missing values in datasets with known marginal totals
- **Matrix completion**: Complete partially observed matrices with constraints
- **Statistical modeling**: Generate synthetic data matching specific row/column distributions
- **Operations research**: Solve transportation and allocation problems
- **Survey data**: Reconstruct contingency tables from partial observations

## Limitations

- The problem may not have an exact solution if constraints are inconsistent
- The algorithm finds the best least-squares approximation when exact solutions don't exist
- Performance may degrade for very large matrices with many unknowns

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License.

## Author

**Shehab Hassani**  
Email: shehab.hassani@areednow.com

## Links

- **Homepage**: [https://github.com/AreedAdmin/matrix-filler.git](https://github.com/AreedAdmin/matrix-filler.git)
- **Source Code**: [https://github.com/AreedAdmin/matrix-filler.git](https://github.com/AreedAdmin/matrix-filler.git)

