\# Day 3 — Indexing, Slicing \& Broadcasting



\## 1. Objective

Master advanced indexing, slicing, boolean masking, fancy indexing, and broadcasting for NumPy-based ML data manipulation.



\## 2. Detailed Tasks

\- Basic 1D and 2D slicing

\- Boolean masking

\- Fancy indexing

\- Broadcasting examples

\- Broadcasting rules documented with worked examples



\## 3. Practical Coding Exercises

Implemented:

\- `filter\_outliers()`

\- `extract\_submatrix()`

\- `normalize\_broadcast()`

\- `select\_rows\_by\_index()`



\## 4. Hands-on Implementation

\- `normalize\_broadcast()` uses NumPy broadcasting without explicit loops.

\- Shape and invalid-input checks are included.

\- Normalization was verified with column mean ≈ 0 and standard deviation ≈ 1.



\## 5. Expected Output / Deliverables

Main deliverable:

\- `day3\_indexing\_broadcasting.py`



Additional task files:

\- `task3\_broadcasting.py`

\- `task4\_practical\_functions.py`

\- `task5\_normalize\_broadcast.py`



\## 6. Testing / Debugging

Created `test\_day3.py` with 8 test cases.



Result:

\- \*\*8/8 tests passed\*\*

\- `np.allclose()` used for normalization validation.

\- Outlier, submatrix, row selection, and invalid-input cases tested.



\## 7. Git / Commit

Day 3 work was committed and pushed to:



`Day3-Indexing-Broadcasting`



Separate task commits were created for the individual implementations and testing.



\## 8. End-of-Day Completion

\- Boolean/fancy indexing implemented

\- Broadcasting demonstrated

\- Broadcasting rules documented

\- `normalize\_broadcast()` uses no explicit loops

\- 8 tests passing

\- Work committed and pushed to GitHub

