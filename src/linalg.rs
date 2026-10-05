use crate::matrix::linear_error;
use pyo3::PyResult;

fn product<const K: usize>(a: &[f64], b: &[f64], ar: usize, ac: usize, br: usize, bc: usize, i: usize, j: usize) -> f64 {
    let mut value = 0.0;
    for k in 0..K {
        value += a[i * ar + k * ac] * b[k * br + j * bc];
    }
    value
}

pub fn multiply(a: &[f64], b: &[f64], out: &mut [f64], rows: usize, inner: usize, cols: usize, ta: bool, tb: bool) {
    let (ar, ac) = if ta { (1, rows) } else { (inner, 1) };
    let (br, bc) = if tb { (1, inner) } else { (cols, 1) };
    for i in 0..rows {
        for j in 0..cols {
            out[i * cols + j] = match inner {
                1 => product::<1>(a, b, ar, ac, br, bc, i, j),
                2 => product::<2>(a, b, ar, ac, br, bc, i, j),
                3 => product::<3>(a, b, ar, ac, br, bc, i, j),
                4 => product::<4>(a, b, ar, ac, br, bc, i, j),
                5 => product::<5>(a, b, ar, ac, br, bc, i, j),
                6 => product::<6>(a, b, ar, ac, br, bc, i, j),
                _ => {
                    let mut value = 0.0;
                    for k in 0..inner {
                        value += a[i * ar + k * ac] * b[k * br + j * bc];
                    }
                    value
                }
            };
        }
    }
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

pub fn quadratic(a: &[f64], h: &[f64]) -> f64 {
    let n = h.len();
    let mut result = 0.0;
    for j in 0..n {
        let mut value = 0.0;
        for i in 0..n {
            value += h[i] * a[i * n + j];
        }
        result += value * h[j];
    }
    result
}

pub fn solve(a: &mut [f64], rhs: &mut [f64], n: usize, cols: usize) -> PyResult<()> {
    for k in 0..n {
        let mut pivot = k;
        for i in k + 1..n {
            if a[i * n + k].abs() > a[pivot * n + k].abs() {
                pivot = i;
            }
        }
        if a[pivot * n + k] == 0.0 {
            return Err(linear_error("Singular matrix"));
        }
        if pivot != k {
            for j in 0..n { a.swap(pivot * n + j, k * n + j); }
            for j in 0..cols { rhs.swap(pivot * cols + j, k * cols + j); }
        }
        for i in k + 1..n {
            let factor = a[i * n + k] / a[k * n + k];
            a[i * n + k] = 0.0;
            for j in k + 1..n { a[i * n + j] = a[i * n + j] - factor * a[k * n + j]; }
            for j in 0..cols { rhs[i * cols + j] = rhs[i * cols + j] - factor * rhs[k * cols + j]; }
        }
    }
    for i in (0..n).rev() {
        for j in 0..cols {
            let mut value = rhs[i * cols + j];
            for k in i + 1..n { value -= a[i * n + k] * rhs[k * cols + j]; }
            rhs[i * cols + j] = value / a[i * n + i];
        }
    }
    Ok(())
}

pub fn cholesky(a: &mut [f64], n: usize) -> PyResult<()> {
    for i in 0..n {
        for j in 0..=i {
            let mut value = a[i * n + j];
            for k in 0..j { value -= a[i * n + k] * a[j * n + k]; }
            if i == j {
                if value <= 0.0 { return Err(linear_error("Matrix is not positive definite")); }
                a[i * n + j] = value.sqrt();
            } else {
                a[i * n + j] = value / a[j * n + j];
            }
        }
        a[i * n + i + 1..(i + 1) * n].fill(0.0);
    }
    Ok(())
}

pub fn lower_solve(a: &[f64], rhs: &mut [f64], n: usize, cols: usize) {
    for i in 0..n {
        for j in 0..cols {
            let mut value = rhs[i * cols + j];
            for k in 0..i { value -= a[i * n + k] * rhs[k * cols + j]; }
            rhs[i * cols + j] = value / a[i * n + i];
        }
    }
}
