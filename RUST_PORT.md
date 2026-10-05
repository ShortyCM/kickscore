# Rust port

Source baseline: `d54428d22e494375971fef3e081a1afc535ab058` on `master`.
Implementation branch: `port/rust-pyo3`.

This implementation has not been built, imported, tested, benchmarked, or numerically validated. No speedup or numerical parity is claimed. Build errors and numerical differences must be resolved using the local stages below before treating this branch as a replacement for the Python implementation.

## Implementation

The PyO3 extension owns the fitters, observations, participant indices, weights, cavity parameters, pseudo-observations, convergence values, kernel parameters, full state histories, and model fit registry. A complete fit call iterates in Rust. The Python objects are API adapters; there is no Python numerical fallback. The Python plotting module is unchanged.

The native implementation includes both fitters, all seven observation types, all five model types, weighted participants, all built-in kernels and kernel sums, prediction before/between/after observations, EP and the existing supported KL paths, likelihood contributions, and incremental addition followed by full-history refitting. Gaussian KL and batch KL likelihood remain unsupported, as in the baseline. PeriodicExponential remains a batch-only kernel; its state-space methods remain unsupported.

Recursive fitting retains the Joseph covariance update and RTS smoothing. Linear systems use partial-pivoted LU; batch fitting uses Cholesky and triangular solves. All arithmetic uses f64. The update order, defaults, learning rates, stopping criterion, quadrature order of 30, stable ordinal formulas, and finite/positive distribution guards are retained. No history truncation, fast-math, reduced precision, or approximate update scheme is used.

Contiguous growable Rust vectors hold persistent histories. Common matrix operations specialize inner dimensions 1 through 6; temporary matrices of up to 64 elements use inline storage. Larger state dimensions retain the general implementation. Observations directly access fitter state, without rebuilding numerical batches or copying their state between fitters and observations. Kernel transition caching is shared across fitters during allocation within a fit. Batch fitting still has quadratic storage and cubic factorization costs. Recursive iterations operate directly on history slices with reusable scratch buffers, including general state dimensions. Batch iterations reuse factorization and multiplication buffers. Prediction and allocation may still create temporary matrices.

The numerical implementation has no BLAS, LAPACK, OpenMP, Rayon, worker threads, or thread pools. Library computations run on the calling thread. Native state objects are thread-affine. NumPy is used for Python input/output conversion and plotting, not fitting or numerical linear algebra.

Fitter and observation arrays expose writable NumPy views of Rust-owned storage. Exported buffers stay alive when an owner is deleted or grows. New pickle round trips retain model connections and fitted state.

Simulation sorts timestamps and samples the built-in Gaussian state-space processes entirely in Rust. It uses a native random generator and covariance eigendecomposition. Gauss-Hermite quadrature retains order 30; nodes are refined using the Hermite recurrence. The modified Bessel function uses a log-domain positive series. These numerical implementations require the comparisons below.

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

These commands install an optimized release extension into the isolated environment. To build a distributable wheel instead, use `python -m maturin build --release --interpreter python` from the activated environment. Wheels are platform-specific. A Windows build does not produce Linux wheels. For a portable Linux wheel, use the manylinux build in the release workflow. The workflows now build native wheels with Maturin; the stale Python dependency lock was removed without resolving or installing dependencies.

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
