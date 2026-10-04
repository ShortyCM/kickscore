use pyo3::{exceptions::PyArithmeticError, prelude::*, types::PyType};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

#[derive(Clone, Serialize, Deserialize)]
pub struct Mat {
    pub n: usize,
    pub m: usize,
    pub v: SmallVec<[f64; 64]>,
}
impl Mat {
    pub fn zero(n: usize, m: usize) -> Self { Self { n, m, v: smallvec::smallvec![0.0; n*m] } }
    pub fn eye(n: usize) -> Self { let mut a = Self::zero(n,n); for i in 0..n { a[(i,i)] = 1.0; } a }
    pub fn from(n: usize, m: usize, v: &[f64]) -> Self { Self { n, m, v: SmallVec::from_slice(v) } }
    pub fn transpose(&self) -> Self { let mut a=Self::zero(self.m,self.n); for i in 0..self.n { for j in 0..self.m { a[(j,i)]=self[(i,j)]; }} a }
    fn dot_fixed<const N:usize>(&self,b:&Self,i:usize,j:usize)->f64 {let mut s=0.0;for k in 0..N{s+=self[(i,k)]*b[(k,j)];}s}
    pub fn mul(&self,b:&Self)->Self {
        let mut a=Self::zero(self.n,b.m);
        for i in 0..self.n {for j in 0..b.m {a[(i,j)]=match self.m {
            1=>self.dot_fixed::<1>(b,i,j),2=>self.dot_fixed::<2>(b,i,j),3=>self.dot_fixed::<3>(b,i,j),4=>self.dot_fixed::<4>(b,i,j),5=>self.dot_fixed::<5>(b,i,j),6=>self.dot_fixed::<6>(b,i,j),
            _=>{let mut s=0.0;for k in 0..self.m{s+=self[(i,k)]*b[(k,j)];}s}
        };}}a
    }
    pub fn add(&self,b:&Self)->Self { let mut a=self.clone(); for (x,y) in a.v.iter_mut().zip(&b.v) { *x+=y; } a }
    pub fn sub(&self,b:&Self)->Self { let mut a=self.clone(); for (x,y) in a.v.iter_mut().zip(&b.v) { *x-=y; } a }
    pub fn scale(&self,s:f64)->Self { let mut a=self.clone(); for x in &mut a.v { *x*=s; } a }
    pub fn quad(&self,h:&Self)->f64 { h.transpose().mul(self).mul(h).v[0] }
    pub fn solve(&self,b:&Self)->PyResult<Self> {
        let n=self.n; let mut a=self.clone(); let mut x=b.clone();
        for k in 0..n {
            let mut p=k; for i in k+1..n { if a[(i,k)].abs()>a[(p,k)].abs() {p=i;} }
            if a[(p,k)]==0.0 || !a[(p,k)].is_finite() { return Err(linear_error("Singular matrix")); }
            if p!=k { for j in 0..n { a.v.swap(p*n+j,k*n+j); } for j in 0..x.m {x.v.swap(p*x.m+j,k*x.m+j);} }
            for i in k+1..n { let f=a[(i,k)]/a[(k,k)]; a[(i,k)]=0.0; for j in k+1..n {a[(i,j)]=a[(i,j)]-f*a[(k,j)];} for j in 0..x.m {x[(i,j)]=x[(i,j)]-f*x[(k,j)];} }
        }
        for i in (0..n).rev() { for j in 0..x.m { for k in i+1..n {x[(i,j)]=x[(i,j)]-a[(i,k)]*x[(k,j)];} x[(i,j)]/=a[(i,i)]; } }
        Ok(x)
    }
    pub fn cholesky(&self)->PyResult<Self> {
        let mut l=Self::zero(self.n,self.n);
        for i in 0..self.n { for j in 0..=i { let mut s=self[(i,j)]; for k in 0..j {s-=l[(i,k)]*l[(j,k)];} if i==j {if s<=0.0 || !s.is_finite() {return Err(linear_error("Matrix is not positive definite"));} l[(i,j)]=s.sqrt();} else {l[(i,j)]=s/l[(j,j)];} }} Ok(l)
    }
    pub fn lower_solve(&self,b:&Self)->Self { let mut x=b.clone(); for i in 0..self.n {for j in 0..b.m {for k in 0..i {x[(i,j)]=x[(i,j)]-self[(i,k)]*x[(k,j)];} x[(i,j)]/=self[(i,i)];}} x }
    pub fn block(parts:&[Self])->Self { let n=parts.iter().map(|a|a.n).sum(); let m=parts.iter().map(|a|a.m).sum(); let mut a=Self::zero(n,m); let(mut r,mut c)=(0,0); for p in parts {for i in 0..p.n {for j in 0..p.m {a[(r+i,c+j)]=p[(i,j)];}} r+=p.n;c+=p.m;} a }
    pub fn rows(&self)->Vec<Vec<f64>> { (0..self.n).map(|i|self.v[i*self.m..(i+1)*self.m].to_vec()).collect() }
}
impl std::ops::Index<(usize,usize)> for Mat {type Output=f64;fn index(&self,(i,j):(usize,usize))->&f64 {&self.v[i*self.m+j]}}
impl std::ops::IndexMut<(usize,usize)> for Mat {fn index_mut(&mut self,(i,j):(usize,usize))->&mut f64 {&mut self.v[i*self.m+j]}}
#[derive(Serialize,Deserialize)]
pub struct History {pub n:usize,pub m:usize,pub v:Vec<f64>}
impl History {
    pub fn new(n:usize,m:usize)->Self{Self{n,m,v:Vec::new()}}
    pub fn push(&mut self,a:Mat){self.v.extend_from_slice(&a.v);}
    pub fn get(&self,i:usize)->Mat{let size=self.n*self.m;Mat::from(self.n,self.m,&self.v[i*size..(i+1)*size])}
    pub fn set(&mut self,i:usize,a:Mat){let size=self.n*self.m;self.v[i*size..(i+1)*size].copy_from_slice(&a.v);}
    pub fn rows(&self)->Vec<Vec<Vec<f64>>>{let size=self.n*self.m;if size==0{return vec![];}(0..self.v.len()/size).map(|i|self.get(i).rows()).collect()}
}

fn linear_error(message:&str)->PyErr {
    Python::attach(|py| {
        match py.import("numpy.linalg").and_then(|m|m.getattr("LinAlgError")).and_then(|t|t.cast_into::<PyType>().map_err(Into::into)) {
            Ok(t)=>PyErr::from_type(t,(message.to_owned(),)),
            Err(_)=>PyArithmeticError::new_err(message.to_owned()),
        }
    })
}
