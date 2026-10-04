# Rust port

Source baseline: `d54428d22e494375971fef3e081a1afc535ab058` on `master`.
Implementation branch: `port/rust-pyo3`.

This implementation has not been built, imported, tested, benchmarked, or numerically validated. No speedup or numerical parity is claimed. Build errors and numerical differences must be resolved using the local stages below before treating this branch as a replacement for the Python implementation.

## Implementation

The PyO3 extension owns the fitters, observations, participant indices, weights, cavity parameters, pseudo-observations, convergence values, kernel parameters, full state histories, and model fit registry. A complete fit call iterates in Rust. The Python objects are API adapters; there is no Python numerical fallback. The Python plotting module is unchanged.

The native implementation includes both fitters, all seven observation types, all five model types, weighted participants, all built-in kernels and kernel sums, prediction before/between/after observations, EP and the existing supported KL paths, likelihood contributions, and incremental addition followed by full-history refitting. Gaussian KL and batch KL likelihood remain unsupported, as in the baseline. PeriodicExponential remains a batch-only kernel; its state-space methods remain unsupported.

Recursive fitting retains the Joseph covariance update and RTS smoothing. Linear systems use partial-pivoted LU; batch fitting uses Cholesky and triangular solves. All arithmetic uses f64. The update order, defaults, learning rates, stopping criterion, quadrature order of 30, stable ordinal formulas, and finite/positive distribution guards are retained. No history truncation, fast-math, reduced precision, or approximate update scheme is used.

Contiguous growable Rust vectors hold persistent histories. Common matrix operations specialize inner dimensions 1 through 6; temporary matrices of up to 64 elements use inline storage. Larger state dimensions retain the general implementation. Observations directly access fitter state, without rebuilding numerical batches or copying their state between fitters and observations. Kernel transition caching is shared across fitters during allocation within a fit. Batch fitting still has quadratic storage and cubic factorization costs. General large matrices still allocate operation results; this is not an allocation-free implementation.

The numerical implementation has no BLAS, LAPACK, OpenMP, Rayon, worker threads, or thread pools. Library computations run on the calling thread. Native state objects are thread-affine. NumPy is used for Python input/output conversion and plotting, not fitting or numerical linear algebra.

## Compatibility and validation boundaries

Fitter score/pseudo-observation arrays, observation participant/cavity arrays, and piecewise kernel bounds expose actual writable NumPy arrays backed by Rust-owned buffers. NumPy slices and `np.asarray` share their storage. An exported buffer remains alive if its owner is deleted or subsequently grows; growing a fitter detaches old exported arrays, as the baseline's concatenation did. The native loop neither copies these arrays through Python nor synchronizes Python numerical objects.

Loading the baseline's Python pickle state is implemented for kernels, both fitters, all built-in observations, and models. It imports allocated/pending samples, posterior state, cavity state, and batch or recursive prediction state without refitting. New pickle round trips also retain shared item/kernel references. Both paths still need the local validation below.

Simulation obtains and restores NumPy's legacy MT19937 state once per call. Random-number generation and the state-space simulation run in Rust. `numpy.random.seed` controls reproducibility. The covariance factorization is computed in Rust, so a seed does not promise bitwise identical samples to NumPy's SVD-based factorization; the covariance and Gaussian distribution are unchanged. Cross-language numerical parity is not yet established.

The baseline supports arbitrary Python computational overrides. Executing those overrides in fitting conflicts with the requirement to execute the complete fitting operation in Rust without Python callbacks. Built-in methods and subclasses that inherit those methods use the native implementation. Overrides that change the computations are explicitly rejected; there is no Python fallback. This follows the explicit native-only, no-Python-callback requirement. It does not omit a built-in model, observation, or kernel, but it is an extension-hook compatibility difference rather than full compatibility with arbitrary user-defined Python code.

The old private ObservationBatch implementation is replaced by native observation storage. Existing tests that specifically assert Python callback counts or monkeypatch Python execution paths do not describe the requested native loop. They have not been removed from the original suite.

Gauss-Hermite nodes/weights are obtained in Rust from the order-30 Jacobi matrix. The modified Bessel function uses a log-domain positive series. These implement the same integrals but do not reproduce SciPy's algorithm bit for bit. Their numerical agreement is specifically covered by the moment stage below and remains unverified.

## Stage 1: build and install

Use a separate checkout and virtual environment. Windows commands below are for Command Prompt, with ordinary CPython 3.14, Rust's stable MSVC toolchain, and the Visual Studio C++ build tools installed. If Rust is missing, install Rustup first and reopen the command prompt. No free-threaded Python build is claimed.

```bat
git clone --branch port/rust-pyo3 https://github.com/ShortyCM/kickscore.git kickscore-rust
cd kickscore-rust
python -m venv .venv-rust
.venv-rust\Scripts\python.exe -m pip install "maturin>=1.9,<2" numpy
set "VIRTUAL_ENV=%CD%\.venv-rust" && .venv-rust\Scripts\python.exe -m maturin develop --release
```

Linux, with Python 3.14, its development headers, a C linker, and stable Rust installed:

```bash
git clone --branch port/rust-pyo3 https://github.com/ShortyCM/kickscore.git kickscore-rust
cd kickscore-rust
python3.14 -m venv .venv-rust
.venv-rust/bin/python -m pip install 'maturin>=1.9,<2' numpy
VIRTUAL_ENV="$PWD/.venv-rust" .venv-rust/bin/python -m maturin develop --release
```

These commands install an optimized release extension into the isolated environment. To build a distributable wheel instead, use `python -m maturin build --release --interpreter python` from the activated environment. Wheels are platform-specific. A Windows build does not produce Linux wheels.

Report the complete build output if it fails, including the first compiler error and its context. If successful, report the final installation lines. Stop there for the first exchange; the following stages are documented for later use.

## Stage 2: existing tests and boundary tests

Install only the validation dependencies into the Rust environment:

```bat
.venv-rust\Scripts\python.exe -m pip install pytest scipy numba
.venv-rust\Scripts\python.exe -m pytest -q tests/test_model.py tests/test_item.py tests/test_observation_ordinal.py tests/test_fitter.py validation/test_native_boundary.py -k "not test_batch_prediction_does_not_build_test_covariance"
```

The excluded test asserts calls to a monkeypatched Python kernel; native batch prediction does not call Python kernels. Run this separately to check the applicable numerical regression tests:

```bat
.venv-rust\Scripts\python.exe -m pytest -q tests/test_inference_regressions.py -k "probit_tie_against_quadrature or logit_extreme_finite_inputs or probit_zero_margin_prediction or zero_variance_cannot_report_convergence or cached_allocation_matches_kernel_at_absolute_times"
.venv-rust\Scripts\python.exe -m pytest -q tests/test_kernel.py tests/test_kernel_equiv.py
```

On Linux replace `.venv-rust\Scripts\python.exe` with `.venv-rust/bin/python`. Report the summary plus the complete first failing traceback. Tests remain unexecuted here.

## Stage 3: compare kernels and likelihood moments

Make an exact baseline checkout and separate environment. These commands do not replace your installed KickScore:

```bat
git worktree add --detach ..\kickscore-python-reference d54428d22e494375971fef3e081a1afc535ab058
python -m venv .venv-reference
.venv-reference\Scripts\python.exe -m pip install numpy scipy numba
.venv-reference\Scripts\python.exe validation\capture.py --source ..\kickscore-python-reference --stage kernels --output reference-kernels.json
.venv-rust\Scripts\python.exe validation\capture.py --source . --stage kernels --output native-kernels.json
.venv-rust\Scripts\python.exe validation\compare.py reference-kernels.json native-kernels.json
```

After resolving any kernel differences:

```bat
.venv-reference\Scripts\python.exe validation\capture.py --source ..\kickscore-python-reference --stage moments --output reference-moments.json
.venv-rust\Scripts\python.exe validation\capture.py --source . --stage moments --output native-moments.json
.venv-rust\Scripts\python.exe validation\compare.py reference-moments.json native-moments.json
```

On Linux use `python3.14`, forward slashes, and `.venv-reference/bin/python` / `.venv-rust/bin/python`. Captures use separate processes and explicit source directories; the Rust package cannot accidentally substitute for the baseline. Report all comparison output. Keep both JSON files for diagnosing differences.

## Stage 4: full-history and incremental model comparison

Only after the earlier stages pass:

```bat
.venv-reference\Scripts\python.exe validation\capture.py --source ..\kickscore-python-reference --stage models --output reference-models.json
.venv-rust\Scripts\python.exe validation\capture.py --source . --stage models --output native-models.json
.venv-rust\Scripts\python.exe validation\compare.py reference-models.json native-models.json
```

This uses small deterministic cases across the model types, supported EP/KL combinations, both fitters, every kernel, Matérn combinations, a general state dimension above eight, equal timestamps, weighted teams, unused items, before/intermediate/after predictions, incremental observations, likelihoods, and pickle round trips. Each fit is four iterations with the same learning rate. Errors are captured and compared as well as numerical results. The comparator uses rtol=2e-10 and atol=2e-11, matching the baseline inference comparison tests; tolerances are not automatically relaxed. Stop on discrepancies and report the comparison output before any larger validation workload.

## Baseline pickle compatibility

After the model comparison passes, create the fixtures with the baseline environment, then load the same files in the Rust environment. Only load these locally generated pickle fixtures.

```bat
.venv-reference\Scripts\python.exe validation\pickle_compatibility.py --source ..\kickscore-python-reference --fixtures legacy-pickles.pkl --output reference-pickles.json --write
.venv-rust\Scripts\python.exe validation\pickle_compatibility.py --source . --fixtures legacy-pickles.pkl --output native-pickles.json
.venv-rust\Scripts\python.exe validation\compare.py reference-pickles.json native-pickles.json
```

This covers both fitters, all model families, unallocated observations, fitted models, pending observations after fitting, prediction state, shared references, and subsequent fitting. Report all comparator output. Use the corresponding `bin/python` paths on Linux.
