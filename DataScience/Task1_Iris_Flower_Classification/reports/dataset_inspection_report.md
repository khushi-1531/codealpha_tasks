# Iris Dataset Inspection Report
**Project:** CodeAlpha Data Science Internship — Task 1 (Iris Flower Classification)  
**Inspection Status:** PASSED

---
## 1. Dataset Dimensions & Schema
- **Total Samples (Rows):** 150
- **Total Attributes (Columns):** 5
- **Column Names:** `sepal_length, sepal_width, petal_length, petal_width, species`

### Data Types
| Column | Data Type | Missing Count |
| :--- | :--- | :--- |
| `sepal_length` | `float64` | 0 |
| `sepal_width` | `float64` | 0 |
| `petal_length` | `float64` | 0 |
| `petal_width` | `float64` | 0 |
| `species` | `str` | 0 |

## 2. Data Integrity
- **Total Missing Values:** 0
- **Duplicate Rows Detected:** 1
  *(Note: 1 duplicate row is expected in the canonical 150-sample Iris dataset at index 142)*

## 3. Target Distribution
| Species | Sample Count | Balance Ratio |
| :--- | :--- | :--- |
| *Iris setosa* | 50 | 33.3% |
| *Iris versicolor* | 50 | 33.3% |
| *Iris virginica* | 50 | 33.3% |

## 4. Descriptive Statistics
| Metric | sepal_length | sepal_width | petal_length | petal_width |
| :--- | :--- | :--- | :--- | :--- |
| **mean** | 5.843 | 3.057 | 3.758 | 1.199 |
| **std** | 0.828 | 0.436 | 1.765 | 0.762 |
| **min** | 4.300 | 2.000 | 1.000 | 0.100 |
| **25%** | 5.100 | 2.800 | 1.600 | 0.300 |
| **50%** | 5.800 | 3.000 | 4.350 | 1.300 |
| **75%** | 6.400 | 3.300 | 5.100 | 1.800 |
| **max** | 7.900 | 4.400 | 6.900 | 2.500 |

## 5. Verification Checklist
- [x] 4 Iris physical measurements present: `True`
- [x] 3 Species represented (setosa, versicolor, virginica): `True`
- [x] Zero missing values: `True`
- [x] Overall structure valid: `True`
