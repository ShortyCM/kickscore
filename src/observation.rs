use crate::storage::{Array,ArrayView};
use crate::{fitter::{FRef,NativeFitter},moments};
use pyo3::{exceptions::{PyFloatingPointError,PyValueError},prelude::*};
use std::{cell::RefCell,rc::Rc};

pub type ORef=Rc<RefCell<OData>>;
pub struct Entry{pub fitter:FRef}
pub struct OData{pub kind:u8,pub p:f64,pub q:f64,pub t:f64,pub entries:Vec<Entry>,pub indices:Array<i64>,pub coeffs:Array<f64>,pub nc:Array<f64>,pub xc:Array<f64>,pub logpart:f64,pub exp_ll:f64}
impl OData{
    pub fn update(&mut self,kl:bool,lr:f64,strict:bool)->PyResult<f64>{
        let(mut m,mut v)=(0.0,0.0);
        for (j,e) in self.entries.iter().enumerate(){let f=e.fitter.borrow();let i=self.indices[j] as usize;if i>=f.vs.len(){return Err(pyo3::exceptions::PyIndexError::new_err("sample not allocated"));}if kl{m+=self.coeffs[j]*f.ms[i];v+=self.coeffs[j]*self.coeffs[j]*f.vs[i];}else{let x=1.0/f.vs[i];self.xc[j]=x-f.xs[i];self.nc[j]=x*f.ms[i]-f.ns[i];if strict && (!self.xc[j].is_finite()||self.xc[j]<=0.0){return Err(PyFloatingPointError::new_err("invalid EP cavity precision"));}m+=self.coeffs[j]*self.nc[j]/self.xc[j];v+=self.coeffs[j]*self.coeffs[j]/self.xc[j];}}
        let(value,first,second)=moments::moments(self.kind,m,v,self.p,self.q,kl)?;
        if strict && !(value.is_finite()&&first.is_finite()&&second.is_finite()){return Err(PyFloatingPointError::new_err("non-finite observation update"));}
        for (j,e) in self.entries.iter().enumerate(){let mut f=e.fitter.borrow_mut();let i=self.indices[j] as usize;let c=self.coeffs[j];let(x,n)=if kl{(-2.0*c*c*second,c*(first-2.0*f.ms[i]*c*second))}else{let den=1.0+c*c*second/self.xc[j];if strict&&(!den.is_finite()||den<=0.0){return Err(PyFloatingPointError::new_err("invalid EP posterior variance"));}(-c*c*second/den,c*(first-c*(self.nc[j]/self.xc[j])*second)/den)};f.xs[i]=(1.0-lr)*f.xs[i]+lr*x;f.ns[i]=(1.0-lr)*f.ns[i]+lr*n;if strict&&!(f.xs[i].is_finite()&&f.ns[i].is_finite()){return Err(PyFloatingPointError::new_err("non-finite pseudo-observation"));}}
        let old=if kl{&mut self.exp_ll}else{&mut self.logpart};let diff=(*old-value).abs();*old=value;Ok(diff)
    }
    pub fn likelihood(&self,kl:bool)->PyResult<f64>{if kl{return Ok(self.exp_ll);}let mut val=self.logpart;for (j,e) in self.entries.iter().enumerate(){let f=e.fitter.borrow();if self.indices[j]<0||self.indices[j] as usize>=f.xs.len(){return Err(pyo3::exceptions::PyIndexError::new_err("sample not allocated"));}let x=f.xs[self.indices[j] as usize];let n=f.ns[self.indices[j] as usize];let argument=x/self.xc[j]+1.0;if argument<=0.0{return Err(PyValueError::new_err("math domain error"));}val+=0.5*argument.ln()+(-n*n-2.0*n*self.nc[j]+x*self.nc[j]*self.nc[j]/self.xc[j])/(2.0*(x+self.xc[j]));}Ok(val)}
}
#[pyclass(unsendable)]
pub struct NativeObservation{pub inner:ORef}
#[pymethods]
impl NativeObservation{
    #[new]
    fn new(fitters:Vec<PyRef<'_,NativeFitter>>,coeffs:Vec<f64>,kind:u8,p:f64,q:f64,t:f64)->PyResult<Self>{
        if fitters.is_empty(){return Err(PyValueError::new_err("need at least one item per observation"));}if fitters.len()!=coeffs.len(){return Err(PyValueError::new_err("participant lengths differ"));}
        let indices=fitters.iter().map(|f|f.inner.borrow_mut().add(t) as i64).collect();
        let entries=fitters.iter().map(|f|Entry{fitter:f.inner.clone()}).collect();
        let n=fitters.len();
        Ok(Self{inner:Rc::new(RefCell::new(OData{kind,p,q,t,entries,indices:Array::from_vec(indices),coeffs:Array::from_vec(coeffs),nc:Array::from_vec(vec![0.0;n]),xc:Array::from_vec(vec![0.0;n]),logpart:0.0,exp_ll:0.0}))})
    }

    fn moments(&self,m:f64,v:f64,kl:bool)->PyResult<(f64,f64,f64)>{let o=self.inner.borrow();moments::moments(o.kind,m,v,o.p,o.q,kl)}
    fn update(&self,kl:bool,lr:f64)->PyResult<f64>{self.inner.borrow_mut().update(kl,lr,false)}
    fn likelihood(&self,kl:bool)->PyResult<f64>{self.inner.borrow().likelihood(kl)}
    fn get_scalar(&self,name:&str)->PyResult<f64>{let o=self.inner.borrow();Ok(match name{"t"=>o.t,"p"=>o.p,"q"=>o.q,"_logpart"=>o.logpart,"_exp_ll"=>o.exp_ll,_=>return Err(PyValueError::new_err("unknown scalar"))})}
    fn set_scalar(&self,name:&str,v:f64)->PyResult<()>{let mut o=self.inner.borrow_mut();match name{"t"=>o.t=v,"p"=>o.p=v,"q"=>o.q=v,"_logpart"=>o.logpart=v,"_exp_ll"=>o.exp_ll=v,_=>return Err(PyValueError::new_err("unknown scalar"))}Ok(())}
    fn buffer(&self,name:&str)->PyResult<ArrayView>{let o=self.inner.borrow();Ok(match name{"_coeffs"=>o.coeffs.view(),"_indices"=>o.indices.view(),"_ns_cav"=>o.nc.view(),"_xs_cav"=>o.xc.view(),_=>return Err(PyValueError::new_err("unknown array"))})}
    fn get_array(&self,name:&str)->PyResult<Vec<f64>>{let o=self.inner.borrow();Ok(match name{"_coeffs"=>o.coeffs.to_vec(),"_indices"=>o.indices.iter().map(|i|*i as f64).collect(),"_ns_cav"=>o.nc.to_vec(),"_xs_cav"=>o.xc.to_vec(),_=>return Err(PyValueError::new_err("unknown array"))})}
    fn set_array(&self,name:&str,v:Vec<f64>)->PyResult<()>{let mut o=self.inner.borrow_mut();if v.len()!=o.entries.len(){return Err(PyValueError::new_err("participant lengths differ"));}match name{"_coeffs"=>o.coeffs=Array::from_vec(v),"_indices"=>o.indices=Array::from_vec(v.iter().map(|x|*x as i64).collect()),"_ns_cav"=>o.nc=Array::from_vec(v),"_xs_cav"=>o.xc=Array::from_vec(v),_=>return Err(PyValueError::new_err("unknown array"))}Ok(())}
    #[staticmethod]
    fn restore(fitters:Vec<PyRef<'_,NativeFitter>>,coeffs:Vec<f64>,indices:Vec<usize>,kind:u8,p:f64,q:f64,t:f64,nc:Vec<f64>,xc:Vec<f64>,logpart:f64,exp_ll:f64)->PyResult<Self>{
        let n=fitters.len();if [coeffs.len(),indices.len(),nc.len(),xc.len()].iter().any(|m|*m!=n){return Err(PyValueError::new_err("participant lengths differ"));}let entries=fitters.iter().map(|f|Entry{fitter:f.inner.clone()}).collect();Ok(Self{inner:Rc::new(RefCell::new(OData{kind,p,q,t,entries,indices:Array::from_vec(indices.iter().map(|i|*i as i64).collect()),coeffs:Array::from_vec(coeffs),nc:Array::from_vec(nc),xc:Array::from_vec(xc),logpart,exp_ll}))})
    }
}
#[pyfunction]
pub fn f_params(fitters:Vec<PyRef<'_,NativeFitter>>,coeffs:Vec<f64>,t:f64)->PyResult<(f64,f64)>{
    if fitters.len()!=coeffs.len(){return Err(PyValueError::new_err("participant lengths differ"));}let(mut m,mut v)=(0.0,0.0);for(f,c)in fitters.iter().zip(coeffs){let(a,b)=f.inner.borrow().predict(t)?;m+=c*a;v+=c*c*b;}Ok((m,v))
}
#[pyclass(unsendable)]
pub struct NativeModel{pub fitters:Vec<FRef>,pub observations:Vec<ORef>,last_kl:Option<bool>,#[pyo3(get,set)]pub parameter:f64,#[pyo3(get,set)]pub last_t:f64}
#[pymethods]
impl NativeModel{
    #[new]
    fn new()->Self{Self{fitters:vec![],observations:vec![],last_kl:None,parameter:0.0,last_t:f64::NEG_INFINITY}}
    fn register_fitter(&mut self,f:PyRef<'_,NativeFitter>){self.fitters.push(f.inner.clone());}
    fn append_observation(&mut self,obs:PyRef<'_,NativeObservation>){self.observations.push(obs.inner.clone());}
    fn get_method(&self)->Option<&str>{self.last_kl.map(|kl|if kl{"kl"}else{"ep"})}
    fn set_method(&mut self,method:Option<&str>)->PyResult<()>{
        self.last_kl=match method {None=>None,Some("ep")=>Some(false),Some("kl")=>Some(true),_=>return Err(PyValueError::new_err("'method' should be one of: 'ep', 'kl'"))};
        Ok(())
    }
    fn fit(&mut self,py:Python<'_>,method:&str,lr:f64,tol:f64,max_iter:isize,verbose:bool)->PyResult<bool>{
        let kl=match method{"ep"=>false,"kl"=>true,_=>return Err(PyValueError::new_err("'method' should be one of: 'ep', 'kl'"))};self.last_kl=Some(kl);
        let result=(||{
            let mut cache=crate::fitter::AllocationCache::default();
            for f in &self.fitters{f.borrow_mut().allocate_cached(&mut cache)?;}
            drop(cache);
            let strict=!self.observations.is_empty() && self.observations.iter().all(|o|o.borrow().kind<4);
            for i in 0..max_iter{
                for f in &self.fitters{let f=f.borrow();if [&f.ms,&f.vs,&f.ns,&f.xs].iter().any(|a|a.iter().any(|v|!v.is_finite()))||f.vs.iter().any(|v|*v<=0.0){return Err(PyFloatingPointError::new_err("invalid score distribution or pseudo-observation"));}}
                let mut diff:f64=0.0;for o in &self.observations{let d=o.borrow_mut().update(kl,lr,strict)?;if !d.is_finite(){return Err(PyFloatingPointError::new_err("non-finite convergence difference"));}diff=diff.max(d);}
                for f in &self.fitters{let mut f=f.borrow_mut();f.fit()?;if f.ms.iter().chain(f.vs.iter()).any(|v|!v.is_finite()){return Err(PyFloatingPointError::new_err("non-finite fitted score distribution"));}if f.vs.iter().any(|v|*v<=0.0){return Err(PyFloatingPointError::new_err("non-positive fitted score variance"));}}
                if verbose{let kwargs=pyo3::types::PyDict::new(py);kwargs.set_item("flush",true)?;py.import("builtins")?.getattr("print")?.call((format!("iteration {}, max diff: {:.5}",i+1,diff),),Some(&kwargs))?;}
                if diff<tol{return Ok(true);}
            }Ok(false)
        })();if result.is_err(){for f in &self.fitters{f.borrow_mut().fitted=false;}}result
    }
    fn likelihood(&self)->PyResult<f64>{let mut val=0.0;for o in &self.observations{val+=o.borrow().likelihood(self.last_kl!=Some(false))?;}for f in &self.fitters{val+=f.borrow().likelihood(self.last_kl!=Some(false))?;}Ok(val)}
    fn outcomes(&self,fitters:Vec<PyRef<'_,NativeFitter>>,coeffs:Vec<f64>,t:f64,kind:u8,p:f64,q:f64,ternary:bool)->PyResult<Vec<f64>>{
        let(m,v)=f_params(fitters,coeffs,t)?;
        let first=moments::probability(kind,m,v,p,q)?;
        if ternary {
            let second=moments::probability(kind+2,m,v,p,q)?;
            Ok(vec![first,second,1.0-first-second])
        } else { Ok(vec![first,1.0-first]) }
    }
    fn count_probabilities(&self,fitters:Vec<PyRef<'_,NativeFitter>>,coeffs:Vec<f64>,t:f64,skellam:bool,base_rate:f64)->PyResult<Vec<f64>>{
        let(m,v)=f_params(fitters,coeffs,t)?;
        if !skellam{let mut out=vec![];let mut sum=0.0;while sum<0.999{let p=moments::probability(5,m,v,out.len() as f64,0.0)?;out.push(p);sum+=p;}Ok(out)}else{let center=moments::probability(6,m,v,0.0,base_rate)?;let(mut neg,mut pos)=(vec![],vec![]);let mut sum=center;while sum<0.999{let k=pos.len() as f64+1.0;let p=moments::probability(6,m,v,k,base_rate)?;let n=moments::probability(6,m,v,-k,base_rate)?;neg.push(n);pos.push(p);sum+=p+n;}neg.reverse();neg.push(center);neg.extend(pos);Ok(neg)}
    }
}
