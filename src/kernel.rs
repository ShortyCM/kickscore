use crate::matrix::Mat;
use crate::storage::{Array,ArrayView};
use pyo3::{exceptions::{PyNotImplementedError, PyValueError}, prelude::*};
use serde::{Deserialize, Serialize};
use std::{cell::RefCell, rc::Rc};

fn scalar_exp(value:f64)->PyResult<f64> {
    let result=value.exp();
    if value.is_finite()&&result==f64::INFINITY {
        return Err(pyo3::exceptions::PyOverflowError::new_err("math range error"));
    }
    Ok(result)
}

pub type KRef=Rc<RefCell<KData>>;
#[derive(Clone, Serialize, Deserialize)]
pub struct KData {pub kind:String,pub p:Vec<f64>,pub bounds:Array<f64>,pub parts:Vec<KRef>}
impl KData {
    pub fn order(&self)->PyResult<usize> {Ok(match self.kind.as_str(){"add"=>self.parts.iter().try_fold(0,|sum,k|Ok::<usize,PyErr>(sum+k.borrow().order()?))?,"affine"|"matern32"=>2,"matern52"=>3,"periodic"=>return Err(PyNotImplementedError::new_err("")),_=>1})}
    pub fn cov(&self,t:f64,s:f64)->f64 {
        let p=&self.p; let r=(t-s).abs();
        match self.kind.as_str(){
            "add"=>self.parts.iter().map(|k|k.borrow().cov(t,s)).sum(),
            "constant"=>p[0],"piecewise"=>if crate::matrix::search_left(&self.bounds,t)==crate::matrix::search_left(&self.bounds,s){p[0]}else{0.0},
            "exponential"=>p[0]*(-r/p[1]).exp(),
            "matern32"=>{let x=3.0_f64.sqrt()*(r/p[1]);p[0]*(1.0+x)*(-x).exp()},
            "matern52"=>{let r=r/p[1];let x=5.0_f64.sqrt()*r;p[0]*(1.0+x+(5.0/3.0)*(r*r))*(-x).exp()},
            "affine"=>p[1]*(t-p[2])*(s-p[2])+p[0],"wiener"=>p[0]*(t.min(s)-p[1])+p[2],
            "periodic"=>p[0]*(-2.0*(r*std::f64::consts::PI/p[2]).sin().abs()/p[1]).exp(),_=>unreachable!()
        }
    }
    pub fn diag(&self,t:f64)->f64 {
        match self.kind.as_str(){
            "add"=>self.parts.iter().map(|k|k.borrow().diag(t)).sum(),
            "affine"=>self.p[1]*(t-self.p[2]).powi(2)+self.p[0],
            "wiener"=>self.p[0]*(t-self.p[1])+self.p[2],
            _=>self.p[0],
        }
    }
    pub fn h(&self)->PyResult<Mat> {let mut h=Mat::zero(self.order()?,1);if self.kind=="add" {let mut offset=0;for k in &self.parts {let v=k.borrow().h()?;h.v[offset..offset+v.n].copy_from_slice(&v.v);offset+=v.n;}} else {h.v[0]=1.0;} Ok(h)}
    pub fn matrix(&self,what:&str,t:f64,s:f64)->PyResult<Mat> {
        if self.kind=="add" {return Ok(Mat::block(&self.parts.iter().map(|k|k.borrow().matrix(what,t,s)).collect::<PyResult<Vec<_>>>()?));}
        let n=self.order()?;let p=&self.p;
        if self.kind=="exponential"&&p[1]==0.0&&["transition","noise","feedback","density"].contains(&what){return Err(pyo3::exceptions::PyZeroDivisionError::new_err("float division by zero"));}
        let d=s-t;let v=p[0];let mut a=Mat::zero(n,n);
        let l=if self.kind=="matern32" || self.kind=="matern52" {p[2]}else{0.0};let z=d*l;
        match what {
            "state"=>match self.kind.as_str(){
                "affine"=>{let x=t-p[2];a=Mat::from(2,2,&[p[1]*x*x+v,p[1]*x,p[1]*x,p[1]]);},
                "wiener"=>a.v[0]=v*(t-p[1])+p[2],
                "matern32"=>a=Mat::from(2,2,&[v,0.0,0.0,v*l*l]),
                "matern52"=>a=Mat::from(3,3,&[v,0.0,-v*l*l/3.0,0.0,v*l*l/3.0,0.0,-v*l*l/3.0,0.0,v*l.powi(4)]),
                _=>a.v[0]=v,
            },
            "transition"=>match self.kind.as_str(){
                "affine"=>a=Mat::from(2,2,&[1.0,d,0.0,1.0]),
                "exponential"=>a.v[0]=scalar_exp(-d/p[1])?,
                "piecewise"=>a.v[0]=if crate::matrix::search_left(&self.bounds,t)==crate::matrix::search_left(&self.bounds,s){1.0}else{0.0},
                "matern32"=>a=Mat::from(2,2,&[z+1.0,d,-d*l*l,1.0-z]).scale(scalar_exp(-z)?),
                "matern52"=>a=Mat::from(3,3,&[z*z/2.0+z+1.0,d*(z+1.0),d*d/2.0,-z*z*l/2.0,-z*z+z+1.0,-d/2.0*(z-2.0),z*l*l/2.0*(z-2.0),z*l*(z-3.0),(z*z-4.0*z+2.0)/2.0]).scale(scalar_exp(-z)?),
                _=>a.v[0]=1.0,
            },
            "noise"=>match self.kind.as_str(){
                "exponential"=>a.v[0]=v*(1.0-scalar_exp(-2.0*d/p[1])?),
                "wiener"=>a.v[0]=v*d,
                "piecewise"=>a.v[0]=if crate::matrix::search_left(&self.bounds,t)==crate::matrix::search_left(&self.bounds,s){0.0}else{v},
                "matern32"=>{let c=scalar_exp(-2.0*z)?;let x=1.0-c*(2.0*z*z+2.0*z+1.0);let y=c*2.0*z*z*l;let w=l*l*(1.0-c*(2.0*z*z-2.0*z+1.0));a=Mat::from(2,2,&[x,y,y,w]).scale(v);},
                "matern52"=>{let c=scalar_exp(-2.0*z)?;let x=-(c*(2.0*z.powi(4)+4.0*z.powi(3)+6.0*z*z+6.0*z+3.0)-3.0)/3.0;let y=c*2.0/3.0*l*z.powi(4);let w=-l*l/3.0*(c*(2.0*z.powi(4)-4.0*z.powi(3)-2.0*z*z-2.0*z-1.0)+1.0);let u=-l*l/3.0*(c*(2.0*z.powi(4)-4.0*z.powi(3)+2.0*z*z+2.0*z+1.0)-1.0);let q=c*2.0/3.0*z*z*l.powi(3)*(z-2.0).powi(2);let r=-l.powi(4)/3.0*(c*(2.0*z.powi(4)-12.0*z.powi(3)+22.0*z*z-10.0*z+3.0)-3.0);a=Mat::from(3,3,&[x,y,w,y,u,q,w,q,r]).scale(v);},_=>{},
            },
            "feedback"=>match self.kind.as_str(){"piecewise"=>return Err(PyNotImplementedError::new_err("")),"exponential"=>a.v[0]=-1.0/p[1],"affine"=>a[(0,1)]=1.0,"matern32"=>a=Mat::from(2,2,&[0.0,1.0,-l*l,-2.0*l]),"matern52"=>a=Mat::from(3,3,&[0.0,1.0,0.0,0.0,0.0,1.0,-l.powi(3),-3.0*l*l,-3.0*l]),_=>{}},
            "effect"=>{if self.kind=="piecewise" {return Err(PyNotImplementedError::new_err(""));} a=Mat::zero(n,1);a[(n-1,0)]=1.0;},
            "density"=>{a=Mat::zero(1,1);a.v[0]=match self.kind.as_str(){"piecewise"=>return Err(PyNotImplementedError::new_err("")),"exponential"=>2.0*v/p[1],"matern32"=>4.0*v*l.powi(3),"matern52"=>16.0/3.0*v*l.powi(5),"wiener"=>v,_=>0.0};},
            _=>return Err(PyValueError::new_err("unknown matrix")),
        } Ok(a)
    }
}
#[pyclass(unsendable)]
#[derive(Clone)]
pub struct NativeKernel {pub inner:KRef}
#[pymethods]
impl NativeKernel {
    #[new]
    fn new(kind:String,mut p:Vec<f64>,mut bounds:Vec<f64>,parts:Vec<PyRef<'_,NativeKernel>>)->PyResult<Self> {
        if !["constant","piecewise","exponential","matern32","matern52","affine","wiener","periodic","add"].contains(&kind.as_str()){return Err(PyValueError::new_err("unknown kernel"));}
        if (kind=="matern32"||kind=="matern52")&&p.len()==2{if p[1]==0.0{return Err(pyo3::exceptions::PyZeroDivisionError::new_err("float division by zero"));}p.push(if kind=="matern32"{3.0_f64.sqrt()/p[1]}else{5.0_f64.sqrt()/p[1]});}
        let needed=match kind.as_str(){"add"=>0,"constant"|"piecewise"=>1,"exponential"=>2,_=>3};
        if p.len()!=needed{return Err(PyValueError::new_err("invalid kernel parameters"));}bounds.sort_by(crate::matrix::time_cmp);
        Ok(Self{inner:Rc::new(RefCell::new(KData{kind,p,bounds:Array::from_vec(bounds),parts:parts.iter().map(|k|k.inner.clone()).collect()}))})
    }
    fn get_param(&self,i:usize)->PyResult<f64>{self.inner.borrow().p.get(i).copied().ok_or_else(||PyValueError::new_err("invalid parameter"))}
    fn set_param(&self,i:usize,v:f64)->PyResult<()>{let mut k=self.inner.borrow_mut();*k.p.get_mut(i).ok_or_else(||PyValueError::new_err("invalid parameter"))?=v;Ok(())}
    fn bounds_buffer(&self)->ArrayView{self.inner.borrow().bounds.view()}
    fn bounds(&self)->Vec<f64>{self.inner.borrow().bounds.to_vec()}
    fn set_bounds(&self,v:Vec<f64>){self.inner.borrow_mut().bounds=Array::from_vec(v);}
    fn order(&self)->PyResult<usize>{self.inner.borrow().order()}
    fn matrix(&self,what:&str,t:f64,s:f64)->PyResult<Vec<Vec<f64>>>{Ok(self.inner.borrow().matrix(what,t,s)?.rows())}
    fn mean(&self)->PyResult<Vec<f64>>{Ok(vec![0.0;self.inner.borrow().order()?])}
    fn h(&self)->PyResult<Vec<f64>>{Ok(self.inner.borrow().h()?.v.to_vec())}
    fn k_mat(&self,a:Vec<f64>,b:Vec<f64>)->Vec<Vec<f64>>{let k=self.inner.borrow();a.iter().map(|t|b.iter().map(|s|k.cov(*t,*s)).collect()).collect()}
    fn k_diag(&self,a:Vec<f64>)->Vec<f64>{let k=self.inner.borrow();a.iter().map(|t|k.diag(*t)).collect()}
    fn simulate(&self,py:Python<'_>,mut ts:Vec<f64>)->PyResult<Vec<f64>>{
        let k=self.inner.borrow();let n=k.order()?;
        ts.sort_by(crate::matrix::time_cmp);if ts.is_empty(){return Err(pyo3::exceptions::PyIndexError::new_err("index 0 is out of bounds for axis 0 with size 0"));}
        let h=k.h()?;let mut x=Mat::zero(n,1);let mut out=Vec::with_capacity(ts.len());let mut rng=crate::random::RandomState::new();
        for (i,t) in ts.iter().enumerate(){let cov=if i==0{k.matrix("state",*t,*t)?}else{x=k.matrix("transition",ts[i-1],*t)?.mul(&x);k.matrix("noise",ts[i-1],*t)?};let (vals,vecs)=crate::moments::eigen(cov)?;if vals.iter().any(|v|*v < -1e-8){py.import("warnings")?.call_method1("warn",("covariance is not symmetric positive-semidefinite.",py.get_type::<pyo3::exceptions::PyRuntimeWarning>()))?;}let z=Mat::from(n,1,&(0..n).map(|j|{let r=rng.normal();r*vals[j].abs().sqrt()}).collect::<Vec<_>>());x=x.add(&vecs.mul(&z));out.push(h.transpose().mul(&x).v[0]);}Ok(out)
    }
}
