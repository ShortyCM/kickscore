use crate::{fitter::{FRef,NativeFitter},moments};
use pyo3::{exceptions::{PyFloatingPointError,PyValueError,PyNotImplementedError},prelude::*};
use std::{cell::RefCell,rc::Rc};

pub type ORef=Rc<RefCell<OData>>;
pub struct Entry{pub fitter:FRef,pub index:usize,pub coeff:f64,pub nc:f64,pub xc:f64}
pub struct OData{pub kind:u8,pub p:f64,pub q:f64,pub t:f64,pub entries:Vec<Entry>,pub logpart:f64,pub exp_ll:f64}
impl OData{
    pub fn update(&mut self,kl:bool,lr:f64,strict:bool)->PyResult<f64>{
        let(mut m,mut v)=(0.0,0.0);
        for e in &mut self.entries{let f=e.fitter.borrow();let i=e.index;if i>=f.vs.len(){return Err(pyo3::exceptions::PyIndexError::new_err("sample not allocated"));}if kl{m+=e.coeff*f.ms[i];v+=e.coeff*e.coeff*f.vs[i];}else{let x=1.0/f.vs[i];e.xc=x-f.xs[i];e.nc=x*f.ms[i]-f.ns[i];if strict && (!e.xc.is_finite()||e.xc<=0.0){return Err(PyFloatingPointError::new_err("invalid EP cavity precision"));}m+=e.coeff*e.nc/e.xc;v+=e.coeff*e.coeff/e.xc;}}
        let(value,first,second)=moments::moments(self.kind,m,v,self.p,self.q,kl)?;
        if strict && !(value.is_finite()&&first.is_finite()&&second.is_finite()){return Err(PyFloatingPointError::new_err("non-finite observation update"));}
        for e in &self.entries{let mut f=e.fitter.borrow_mut();let i=e.index;let c=e.coeff;let(x,n)=if kl{(-2.0*c*c*second,c*(first-2.0*f.ms[i]*c*second))}else{let den=1.0+c*c*second/e.xc;if strict&&(!den.is_finite()||den<=0.0){return Err(PyFloatingPointError::new_err("invalid EP posterior variance"));}(-c*c*second/den,c*(first-c*e.nc/e.xc*second)/den)};f.xs[i]=(1.0-lr)*f.xs[i]+lr*x;f.ns[i]=(1.0-lr)*f.ns[i]+lr*n;if strict&&!(f.xs[i].is_finite()&&f.ns[i].is_finite()){return Err(PyFloatingPointError::new_err("non-finite pseudo-observation"));}}
        let old=if kl{&mut self.exp_ll}else{&mut self.logpart};let diff=(*old-value).abs();*old=value;Ok(diff)
    }
    pub fn likelihood(&self,kl:bool)->f64{if kl{return self.exp_ll;}let mut val=self.logpart;for e in &self.entries{let f=e.fitter.borrow();let x=f.xs[e.index];let n=f.ns[e.index];val+=0.5*(x/e.xc+1.0).ln()+(-n*n-2.0*n*e.nc+x*e.nc*e.nc/e.xc)/(2.0*(x+e.xc));}val}
}
#[pyclass(unsendable)]
pub struct NativeObservation{pub inner:ORef}
#[pymethods]
impl NativeObservation{
    #[new]
    fn new(fitters:Vec<PyRef<'_,NativeFitter>>,coeffs:Vec<f64>,kind:u8,p:f64,q:f64,t:f64)->PyResult<Self>{
        if fitters.is_empty(){return Err(PyValueError::new_err("need at least one item per observation"));}if fitters.len()!=coeffs.len(){return Err(PyValueError::new_err("participant lengths differ"));}
        let entries=fitters.iter().zip(coeffs).map(|(f,c)|Entry{index:f.inner.borrow_mut().add(t),fitter:f.inner.clone(),coeff:c,nc:0.0,xc:0.0}).collect();Ok(Self{inner:Rc::new(RefCell::new(OData{kind,p,q,t,entries,logpart:0.0,exp_ll:0.0}))})
    }
    fn moments(&self,m:f64,v:f64,kl:bool)->PyResult<(f64,f64,f64)>{let o=self.inner.borrow();moments::moments(o.kind,m,v,o.p,o.q,kl)}
    fn update(&self,kl:bool,lr:f64)->PyResult<f64>{self.inner.borrow_mut().update(kl,lr,false)}
    fn likelihood(&self,kl:bool)->f64{self.inner.borrow().likelihood(kl)}
    fn get_scalar(&self,name:&str)->PyResult<f64>{let o=self.inner.borrow();Ok(match name{"t"=>o.t,"p"=>o.p,"q"=>o.q,"_logpart"=>o.logpart,"_exp_ll"=>o.exp_ll,_=>return Err(PyValueError::new_err("unknown scalar"))})}
    fn set_scalar(&self,name:&str,v:f64)->PyResult<()>{let mut o=self.inner.borrow_mut();match name{"t"=>o.t=v,"p"=>o.p=v,"q"=>o.q=v,"_logpart"=>o.logpart=v,"_exp_ll"=>o.exp_ll=v,_=>return Err(PyValueError::new_err("unknown scalar"))}Ok(())}
    fn get_array(&self,name:&str)->PyResult<Vec<f64>>{let o=self.inner.borrow();o.entries.iter().map(|e|Ok(match name{"_coeffs"=>e.coeff,"_indices"=>e.index as f64,"_ns_cav"=>e.nc,"_xs_cav"=>e.xc,_=>return Err(PyValueError::new_err("unknown array"))})).collect()}
    fn set_array(&self,name:&str,v:Vec<f64>)->PyResult<()>{let mut o=self.inner.borrow_mut();if v.len()!=o.entries.len(){return Err(PyValueError::new_err("participant lengths differ"));}for(e,v)in o.entries.iter_mut().zip(v){match name{"_coeffs"=>e.coeff=v,"_indices"=>e.index=v as usize,"_ns_cav"=>e.nc=v,"_xs_cav"=>e.xc=v,_=>return Err(PyValueError::new_err("unknown array"))}}Ok(())}
    #[staticmethod]
    fn restore(fitters:Vec<PyRef<'_,NativeFitter>>,coeffs:Vec<f64>,indices:Vec<usize>,kind:u8,p:f64,q:f64,t:f64,nc:Vec<f64>,xc:Vec<f64>,logpart:f64,exp_ll:f64)->PyResult<Self>{
        let n=fitters.len();if [coeffs.len(),indices.len(),nc.len(),xc.len()].iter().any(|m|*m!=n){return Err(PyValueError::new_err("participant lengths differ"));}let entries=(0..n).map(|i|Entry{fitter:fitters[i].inner.clone(),index:indices[i],coeff:coeffs[i],nc:nc[i],xc:xc[i]}).collect();Ok(Self{inner:Rc::new(RefCell::new(OData{kind,p,q,t,entries,logpart,exp_ll}))})
    }
}
#[pyfunction]
pub fn f_params(fitters:Vec<PyRef<'_,NativeFitter>>,coeffs:Vec<f64>,t:f64)->PyResult<(f64,f64)>{
    if fitters.len()!=coeffs.len(){return Err(PyValueError::new_err("participant lengths differ"));}let(mut m,mut v)=(0.0,0.0);for(f,c)in fitters.iter().zip(coeffs){let(a,b)=f.inner.borrow().predict(t)?;m+=c*a;v+=c*c*b;}Ok((m,v))
}
#[pyclass(unsendable)]
pub struct NativeModel{pub fitters:Vec<FRef>,pub observations:Vec<ORef>,last_kl:bool,#[pyo3(get,set)]pub last_t:f64}
#[pymethods]
impl NativeModel{
    #[new]
    fn new()->Self{Self{fitters:vec![],observations:vec![],last_kl:true,last_t:f64::NEG_INFINITY}}
    fn register_fitter(&mut self,f:PyRef<'_,NativeFitter>){self.fitters.push(f.inner.clone());}
    fn set_observations(&mut self,obs:Vec<PyRef<'_,NativeObservation>>){self.observations=obs.iter().map(|o|o.inner.clone()).collect();}
    fn set_method(&mut self,kl:bool){self.last_kl=kl;}
    fn fit(&mut self,py:Python<'_>,method:&str,lr:f64,tol:f64,max_iter:isize,verbose:bool)->PyResult<bool>{
        let kl=match method{"ep"=>false,"kl"=>true,_=>return Err(PyValueError::new_err("'method' should be one of: 'ep', 'kl'"))};self.last_kl=kl;
        let result=(||{
            let mut cache=std::collections::HashMap::new();
            for f in &self.fitters{f.borrow_mut().allocate_cached(&mut cache)?;}
            drop(cache);
            let strict=!self.observations.is_empty() && self.observations.iter().all(|o|o.borrow().kind<4);
            for i in 0..max_iter{
                for f in &self.fitters{let f=f.borrow();if [&f.ms,&f.vs,&f.ns,&f.xs].iter().any(|a|a.iter().any(|v|!v.is_finite()))||f.vs.iter().any(|v|*v<=0.0){return Err(PyFloatingPointError::new_err("invalid score distribution or pseudo-observation"));}}
                let mut diff:f64=0.0;for o in &self.observations{let d=o.borrow_mut().update(kl,lr,strict)?;if !d.is_finite(){return Err(PyFloatingPointError::new_err("non-finite convergence difference"));}diff=diff.max(d);}
                for f in &self.fitters{let mut f=f.borrow_mut();f.fit()?;if f.ms.iter().chain(&f.vs).any(|v|!v.is_finite()){return Err(PyFloatingPointError::new_err("non-finite fitted score distribution"));}if f.vs.iter().any(|v|*v<=0.0){return Err(PyFloatingPointError::new_err("non-positive fitted score variance"));}}
                if verbose{let kwargs=pyo3::types::PyDict::new(py);kwargs.set_item("flush",true)?;py.import("builtins")?.getattr("print")?.call((format!("iteration {}, max diff: {:.5}",i+1,diff),),Some(&kwargs))?;}
                if diff<tol{return Ok(true);}
            }Ok(false)
        })();if result.is_err(){for f in &self.fitters{f.borrow_mut().fitted=false;}}result
    }
    fn likelihood(&self)->PyResult<f64>{let mut val=0.0;for o in &self.observations{val+=o.borrow().likelihood(self.last_kl);}for f in &self.fitters{val+=f.borrow().likelihood(self.last_kl)?;}Ok(val)}
    fn reject_override(&self)->PyResult<()>{for f in &self.fitters{f.borrow_mut().fitted=false;}Err(PyNotImplementedError::new_err("Python overrides of core computations cannot run in the Rust fitting loop"))}
    fn count_probabilities(&self,fitters:Vec<PyRef<'_,NativeFitter>>,coeffs:Vec<f64>,t:f64,skellam:bool,base_rate:f64)->PyResult<Vec<f64>>{
        let(m,v)=f_params(fitters,coeffs,t)?;
        if !skellam{let mut out=vec![];let mut sum=0.0;while sum<0.999{let p=moments::probability(5,m,v,out.len() as f64,0.0)?;out.push(p);sum+=p;}Ok(out)}else{let center=moments::probability(6,m,v,0.0,base_rate)?;let(mut neg,mut pos)=(vec![],vec![]);let mut sum=center;while sum<0.999{let k=pos.len() as f64+1.0;let p=moments::probability(6,m,v,k,base_rate)?;let n=moments::probability(6,m,v,-k,base_rate)?;neg.push(n);pos.push(p);sum+=p+n;}neg.reverse();neg.push(center);neg.extend(pos);Ok(neg)}
    }
}
