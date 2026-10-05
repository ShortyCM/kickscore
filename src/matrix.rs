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
    pub fn resize(&mut self,n:usize,m:usize){self.n=n;self.m=m;self.v.resize(n*m,0.0);}

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
    pub fn quad(&self,h:&Self)->f64 { crate::linalg::quadratic(&self.v,&h.v) }
    pub fn solve(&self,b:&Self)->PyResult<Self> {
        let mut a=self.clone();
        let mut x=b.clone();
        crate::linalg::solve(&mut a.v,&mut x.v,self.n,b.m)?;
        Ok(x)
    }
    pub fn block(parts:&[Self])->Self { let n=parts.iter().map(|a|a.n).sum(); let m=parts.iter().map(|a|a.m).sum(); let mut a=Self::zero(n,m); let(mut r,mut c)=(0,0); for p in parts {for i in 0..p.n {for j in 0..p.m {a[(r+i,c+j)]=p[(i,j)];}} r+=p.n;c+=p.m;} a }
    pub fn rows(&self)->Vec<Vec<f64>> { (0..self.n).map(|i|self.v[i*self.m..(i+1)*self.m].to_vec()).collect() }
}
impl std::ops::Index<(usize,usize)> for Mat {type Output=f64;fn index(&self,(i,j):(usize,usize))->&f64 {&self.v[i*self.m+j]}}
impl std::ops::IndexMut<(usize,usize)> for Mat {fn index_mut(&mut self,(i,j):(usize,usize))->&mut f64 {&mut self.v[i*self.m+j]}}
#[derive(Serialize,Deserialize)]
pub struct History {pub n:usize,pub m:usize,pub v:Vec<f64>}
impl History {
    pub fn new(n:usize,m:usize)->Self{Self{n,m,v:Vec::new()}}
    pub fn push(&mut self,a:&Mat){self.v.extend_from_slice(&a.v);}
    pub fn push_zero(&mut self){self.v.resize(self.v.len()+self.n*self.m,0.0);}
    pub fn get(&self,i:usize)->Mat{let size=self.n*self.m;Mat::from(self.n,self.m,&self.v[i*size..(i+1)*size])}
    pub fn rows(&self)->Vec<Vec<Vec<f64>>>{let size=self.n*self.m;if size==0{return vec![];}(0..self.v.len()/size).map(|i|self.get(i).rows()).collect()}
}

pub fn linear_error(message:&str)->PyErr {
    Python::attach(|py| {
        match py.import("numpy.linalg").and_then(|m|m.getattr("LinAlgError")).and_then(|t|t.cast_into::<PyType>().map_err(Into::into)) {
            Ok(t)=>PyErr::from_type(t,(message.to_owned(),)),
            Err(_)=>PyArithmeticError::new_err(message.to_owned()),
        }
    })
}

pub fn time_cmp(a:&f64,b:&f64)->std::cmp::Ordering {
    match a.partial_cmp(b) {
        Some(order)=>order,
        None=>match (a.is_nan(),b.is_nan()) {
            (true,true)=>std::cmp::Ordering::Equal,
            (true,false)=>std::cmp::Ordering::Greater,
            _=>std::cmp::Ordering::Less,
        },
    }
}

pub fn search_left(values:&[f64],value:f64)->usize {
    values.partition_point(|entry|time_cmp(entry,&value)==std::cmp::Ordering::Less)
}
