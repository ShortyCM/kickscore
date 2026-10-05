use crate::{fitter::FData, linalg::{cholesky, dot, lower_solve, multiply, quadratic, solve}};
use pyo3::PyResult;

#[derive(Default)]
pub struct Workspace {
    first: Vec<f64>,
    second: Vec<f64>,
    third: Vec<f64>,
    fourth: Vec<f64>,
    vector: Vec<f64>,
    gain: Vec<f64>,
}

impl Workspace {
    fn prepare(&mut self, n: usize) {
        self.first.resize(n * n, 0.0);
        self.second.resize(n * n, 0.0);
        self.third.resize(n * n, 0.0);
        self.fourth.resize(n * n, 0.0);
        self.vector.resize(n, 0.0);
        self.gain.resize(n, 0.0);
    }
}

pub fn recursive(f: &mut FData) -> PyResult<()> {
    let count = f.ts.len();
    let n = f.h.n;
    let size = n * n;
    f.work.prepare(n);
    let Workspace { first, second, third, fourth, vector, gain } = &mut f.work;
    let h = &f.h.v;
    for i in 0..count {
        let vr = i * n..(i + 1) * n;
        let mr = i * size..(i + 1) * size;
        if i > 0 {
            let a = &f.a.v[(i - 1) * size..i * size];
            multiply(a, &f.mf.v[(i - 1) * n..i * n], &mut f.mp.v[vr.clone()], n, n, 1, false, false);
            multiply(a, &f.pf.v[(i - 1) * size..i * size], first, n, n, n, false, false);
            multiply(first, a, &mut f.pp.v[mr.clone()], n, n, n, false, true);
            for j in 0..size { f.pp.v[i * size + j] += f.q.v[(i - 1) * size + j]; }
        }
        let p = &f.pp.v[mr.clone()];
        let denom = 1.0 + f.xs[i] * quadratic(p, h);
        multiply(p, h, gain, n, n, 1, false, false);
        let residual = f.ns[i] - f.xs[i] * dot(h, &f.mp.v[vr.clone()]);
        for j in 0..n {
            gain[j] /= denom;
            f.mf.v[i * n + j] = f.mp.v[i * n + j] + gain[j] * residual;
            for k in 0..n {
                second[j * n + k] = (if j == k { 1.0 } else { 0.0 }) - f.xs[i] * gain[j] * h[k];
            }
        }
        multiply(second, p, first, n, n, n, false, false);
        multiply(first, second, &mut f.pf.v[mr], n, n, n, false, true);
        for j in 0..n {
            for k in 0..n { f.pf.v[i * size + j * n + k] += f.xs[i] * (gain[j] * gain[k]); }
        }
    }
    for i in (0..count).rev() {
        let vr = i * n..(i + 1) * n;
        let mr = i * size..(i + 1) * size;
        if i + 1 == count {
            f.sm.v[vr.clone()].copy_from_slice(&f.mf.v[vr.clone()]);
            f.sp.v[mr.clone()].copy_from_slice(&f.pf.v[mr.clone()]);
        } else {
            multiply(&f.a.v[mr.clone()], &f.pf.v[mr.clone()], first, n, n, n, false, false);
            second.copy_from_slice(&f.pp.v[(i + 1) * size..(i + 2) * size]);
            solve(second, first, n, n)?;
            for j in 0..n {
                vector[j] = f.sm.v[(i + 1) * n + j] - f.mp.v[(i + 1) * n + j];
                for k in 0..n { third[j * n + k] = first[k * n + j]; }
            }
            multiply(third, vector, gain, n, n, 1, false, false);
            for j in 0..n { f.sm.v[i * n + j] = f.mf.v[i * n + j] + gain[j]; }
            for j in 0..size { fourth[j] = f.sp.v[(i + 1) * size + j] - f.pp.v[(i + 1) * size + j]; }
            multiply(third, fourth, first, n, n, n, false, false);
            multiply(first, third, second, n, n, n, false, true);
            for j in 0..size { f.sp.v[i * size + j] = f.pf.v[i * size + j] + second[j]; }
        }
        f.ms[i] = dot(h, &f.sm.v[vr]);
        f.vs[i] = quadratic(&f.sp.v[mr], h);
    }
    Ok(())
}

pub fn batch(f: &mut FData) -> PyResult<()> {
    let n = f.ts.len();
    f.work.first.resize(n*n,0.0);
    f.work.second.resize(n*n,0.0);
    f.work.vector.resize(n,0.0);
    f.chol.resize(n, n);
    f.wi.resize(n, n);
    f.cov.resize(n, n);
    f.wv.resize(n, 1);
    f.ms.prepare_output(n);
    f.vs.prepare_output(n);
    let Workspace { first, second, vector, .. } = &mut f.work;
    for i in 0..n { vector[i] = f.xs[i].sqrt(); }
    for i in 0..n {
        for j in 0..n {
            f.chol.v[i * n + j] = vector[i] * vector[j] * f.km.v[i * n + j] + if i == j { 1.0 } else { 0.0 };
            first[i * n + j] = if i == j { vector[i] } else { 0.0 };
        }
    }
    cholesky(&mut f.chol.v, n)?;
    if f.chol.v.iter().chain(first.iter()).any(|x|!x.is_finite()){return Err(pyo3::exceptions::PyValueError::new_err("array must not contain infs or NaNs"));}
    lower_solve(&f.chol.v, first, n, n);
    multiply(first, first, &mut f.wi.v, n, n, n, true, false);
    multiply(&f.km.v, &f.wi.v, first, n, n, n, false, false);
    multiply(first, &f.km.v, second, n, n, n, false, false);
    for i in 0..n * n { f.cov.v[i] = f.km.v[i] - second[i]; }
    multiply(&f.cov.v, &f.ns, &mut f.ms, n, n, 1, false, false);
    for i in 0..n { f.vs[i] = f.cov.v[i * n + i]; }
    multiply(&f.wi.v, &f.km.v, first, n, n, n, false, false);
    multiply(first, &f.ns, &mut f.wv.v, n, n, 1, false, false);
    for i in 0..n { f.wv.v[i] = f.ns[i] - f.wv.v[i]; }
    Ok(())
}
