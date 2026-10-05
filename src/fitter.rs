use crate::storage::{Array,ArrayView};
use crate::{kernel::{KRef,NativeKernel},matrix::{Mat,History}};
use pyo3::{exceptions::{PyRuntimeError,PyValueError,PyNotImplementedError},prelude::*};
use serde::{Deserialize,Serialize};
use std::{cell::RefCell,rc::Rc,collections::HashMap};

pub type FRef=Rc<RefCell<FData>>;
#[derive(Default)]
pub struct AllocationCache {
    prior:HashMap<usize,Mat>,
    steps:HashMap<(usize,u64),(Mat,Mat)>,
}
#[derive(Serialize,Deserialize)]
pub struct FData{
    pub kernel:KRef,pub batch:bool,pub fitted:bool,pub ts:Array<f64>,pub pending:Array<f64>,pub ms:Array<f64>,pub vs:Array<f64>,pub ns:Array<f64>,pub xs:Array<f64>,
    pub h:Mat,pub a:History,pub q:History,pub mp:History,pub pp:History,pub mf:History,pub pf:History,pub sm:History,pub sp:History,
    #[serde(skip)]
    pub work:crate::inference::Workspace,
    pub km:Mat,pub cov:Mat,pub chol:Mat,pub wi:Mat,pub wv:Mat,
}
impl FData{
    pub fn new(kernel:KRef,batch:bool)->PyResult<Self>{let h=if batch{Mat::zero(0,1)}else{kernel.borrow().h()?};let n=h.n;Ok(Self{kernel,batch,fitted:true,ts:Array::new(),pending:Array::new(),ms:Array::new(),vs:Array::new(),ns:Array::new(),xs:Array::new(),h,a:History::new(n,n),q:History::new(n,n),mp:History::new(n,1),pp:History::new(n,n),mf:History::new(n,1),pf:History::new(n,n),sm:History::new(n,1),sp:History::new(n,n),work:crate::inference::Workspace::default(),km:Mat::zero(0,0),cov:Mat::zero(0,0),chol:Mat::zero(0,0),wi:Mat::zero(0,0),wv:Mat::zero(0,1)})}
    pub fn add(&mut self,t:f64)->usize{let i=self.ts.len()+self.pending.len();self.pending.push(t);self.fitted=false;i}
    pub fn allocate(&mut self)->PyResult<()>{self.allocate_cached(&mut AllocationCache::default())}
    pub fn allocate_cached(&mut self,cache:&mut AllocationCache)->PyResult<()> {
        if self.pending.is_empty()&&!self.batch {return Ok(());}
        let additional=self.pending.len();
        for values in [&mut self.ts,&mut self.ms,&mut self.vs,&mut self.ns,&mut self.xs] {
            values.reserve(additional);
        }
        let k=self.kernel.borrow();
        let stationary=fn_stationary(&k);
        let kernel_key=Rc::as_ptr(&self.kernel) as usize;
        if !self.batch {
            for history in [&mut self.a,&mut self.q,&mut self.mp,&mut self.pp,&mut self.mf,&mut self.pf,&mut self.sm,&mut self.sp] {
                history.v.reserve(additional*history.n*history.m);
            }
            if stationary&&!cache.prior.contains_key(&kernel_key) {
                cache.prior.insert(kernel_key,k.matrix("state",self.pending[0],self.pending[0])?);
            }
        }
        for &t in self.pending.iter() {
            self.ms.push(0.0);
            self.vs.push(k.diag(t));
            self.ns.push(0.0);
            self.xs.push(0.0);
            if !self.batch {
                let prior;
                let p=if stationary {&cache.prior[&kernel_key]} else {
                    prior=k.matrix("state",t,t)?;
                    &prior
                };
                self.mp.push_zero();
                self.mf.push_zero();
                self.sm.push_zero();
                self.pp.push(p);
                self.pf.push(p);
                self.sp.push(p);
                if let Some(&prev)=self.ts.last() {
                    if stationary {
                        let key=(kernel_key,(t-prev).to_bits());
                        if !cache.steps.contains_key(&key) {
                            let step=(k.matrix("transition",prev,t)?,k.matrix("noise",prev,t)?);
                            if cache.steps.len()>=1024 {cache.steps.clear();}
                            cache.steps.insert(key,step);
                        }
                        let(a,q)=&cache.steps[&key];
                        self.a.push(a);
                        self.q.push(q);
                    } else {
                        self.a.push(&k.matrix("transition",prev,t)?);
                        self.q.push(&k.matrix("noise",prev,t)?);
                    }
                }
            }
            self.ts.push(t);
        }
        self.pending.clear();
        if self.batch {
            let n=self.ts.len();
            self.km.resize(n,n);
            for i in 0..n {for j in 0..n {self.km[(i,j)]=k.cov(self.ts[i],self.ts[j]);}}
        }
        Ok(())
    }
    pub fn fit(&mut self)->PyResult<()>{
        if !self.pending.is_empty(){return Err(PyRuntimeError::new_err("new data since last call to `allocate()`"));}
        let n=self.ts.len();if n==0{self.fitted=true;return Ok(());}
        if self.batch { crate::inference::batch(self)?; }
        else { crate::inference::recursive(self)?; }
        self.fitted=true;Ok(())
    }
    pub fn ready(&self)->PyResult<()>{if !self.fitted{Err(PyRuntimeError::new_err("new data since last call to `fit()`"))}else{Ok(())}}
    pub fn predict(&self,t:f64)->PyResult<(f64,f64)>{
        self.ready()?;let k=self.kernel.borrow();let n=self.ts.len();if n==0{return Ok((0.0,k.diag(t)));}
        if self.batch{let row=Mat::from(1,n,&self.ts.iter().map(|s|k.cov(t,*s)).collect::<Vec<_>>());return Ok((row.mul(&self.wv).v[0],k.diag(t)-row.mul(&self.wi).mul(&row.transpose()).v[0]));}
        let nxt=crate::matrix::search_left(&self.ts,t);let h=&self.h;let ht=h.transpose();
        if nxt==n{let a=k.matrix("transition",self.ts[n-1],t)?;let q=k.matrix("noise",self.ts[n-1],t)?;return Ok((ht.mul(&a.mul(&self.sm.get(n-1))).v[0],a.mul(&self.sp.get(n-1)).mul(&a.transpose()).add(&q).quad(h)));}
        let(m,p)=if nxt==0{(Mat::zero(h.n,1),k.matrix("state",t,t)?)}else{let a=k.matrix("transition",self.ts[nxt-1],t)?;let q=k.matrix("noise",self.ts[nxt-1],t)?;(a.mul(&self.mf.get(nxt-1)),a.mul(&self.pf.get(nxt-1)).mul(&a.transpose()).add(&q))};
        let a=k.matrix("transition",t,self.ts[nxt])?;let g=self.pp.get(nxt).solve(&a.mul(&p))?.transpose();Ok((ht.mul(&m.add(&g.mul(&self.sm.get(nxt).sub(&self.mp.get(nxt))))).v[0],p.add(&g.mul(&self.sp.get(nxt).sub(&self.pp.get(nxt))).mul(&g.transpose())).quad(h)))
    }
    pub fn likelihood(&self,kl:bool)->PyResult<f64>{
        if kl && self.batch{return Err(PyNotImplementedError::new_err(""));}self.ready()?;let n=self.ts.len();if n==0{return Ok(0.0);}
        if self.batch {
            let mut quadratic=0.0;
            for i in 0..n {
                let mut value=0.0;
                for j in 0..n {value+=self.cov[(i,j)]*self.ns[j];}
                quadratic+=self.ns[i]*value;
            }
            return Ok(-(0..n).map(|i|self.chol[(i,i)].ln()).sum::<f64>()+0.5*quadratic);
        }
        let mut val=0.0;
        let dim=self.h.n;
        for i in 0..n {
            let m=crate::linalg::dot(&self.h.v,&self.mp.v[i*dim..(i+1)*dim]);
            let v=crate::linalg::quadratic(&self.pp.v[i*dim*dim..(i+1)*dim*dim],&self.h.v);
            let x=self.xs[i];let u=self.ns[i];let argument=x*v+1.0;
            if argument<=0.0{return Err(PyValueError::new_err("math domain error"));}
            val+=if kl {
                let ms=crate::linalg::dot(&self.h.v,&self.sm.v[i*dim..(i+1)*dim]);
                let vs=crate::linalg::quadratic(&self.sp.v[i*dim*dim..(i+1)*dim*dim],&self.h.v);
                -0.5*(argument.ln()+x*(m*m-ms*ms-vs)-2.0*u*(m-ms)-(x*m-u).powi(2)/(1.0/v+x))
            } else {-0.5*(argument.ln()+(-u*u*v-2.0*u*m+x*m*m)/argument)};
        }
        Ok(val)
    }
    pub fn array(&self,name:&str)->PyResult<&Vec<f64>>{Ok(match name{"ts"=>&self.ts,"ts_new"=>&self.pending,"ms"=>&self.ms,"vs"=>&self.vs,"ns"=>&self.ns,"xs"=>&self.xs,_=>return Err(PyValueError::new_err("unknown array"))})}
}
fn fn_stationary(k:&crate::kernel::KData)->bool{match k.kind.as_str(){"constant"|"exponential"|"matern32"|"matern52"=>true,"add"=>k.parts.iter().all(|p|fn_stationary(&p.borrow())),_=>false}}
#[pyclass(unsendable)]
pub struct NativeFitter{pub inner:FRef}
#[pymethods]
impl NativeFitter{
    #[new]
    fn new(kernel:PyRef<'_,NativeKernel>,batch:bool)->PyResult<Self>{Ok(Self{inner:Rc::new(RefCell::new(FData::new(kernel.inner.clone(),batch)?))})}
    fn append_pending(&self,t:f64){self.inner.borrow_mut().pending.push(t);}
    fn add_sample(&self,t:f64)->usize{self.inner.borrow_mut().add(t)}
    fn allocate(&self)->PyResult<()>{self.inner.borrow_mut().allocate()}
    fn fit(&self)->PyResult<()>{self.inner.borrow_mut().fit()}
    fn ready(&self)->PyResult<()>{self.inner.borrow().ready()}
    fn is_allocated(&self)->bool{self.inner.borrow().pending.is_empty()}
    fn is_fitted(&self)->bool{self.inner.borrow().fitted}
    fn set_fitted(&self,v:bool){self.inner.borrow_mut().fitted=v;}
    fn buffer(&self,name:&str)->PyResult<ArrayView>{let f=self.inner.borrow();Ok(match name{"ts"=>f.ts.view(),"ms"=>f.ms.view(),"vs"=>f.vs.view(),"ns"=>f.ns.view(),"xs"=>f.xs.view(),_=>return Err(PyValueError::new_err("unknown array"))})}
    fn get_array(&self,name:&str)->PyResult<Vec<f64>>{Ok(self.inner.borrow().array(name)?.clone())}
    fn set_array(&self,name:&str,v:Vec<f64>)->PyResult<()>{let mut f=self.inner.borrow_mut();if name!="ts_new" && v.len()!=f.ts.len(){return Err(PyValueError::new_err("array length must match allocated samples"));}match name{"ts"=>f.ts=Array::from_vec(v),"ts_new"=>f.pending=Array::from_vec(v),"ms"=>f.ms=Array::from_vec(v),"vs"=>f.vs=Array::from_vec(v),"ns"=>f.ns=Array::from_vec(v),"xs"=>f.xs=Array::from_vec(v),_=>return Err(PyValueError::new_err("unknown array"))}Ok(())}
    fn predict(&self,ts:Vec<f64>)->PyResult<(Vec<f64>,Vec<f64>)>{let f=self.inner.borrow();f.ready()?;let mut m=Vec::with_capacity(ts.len());let mut v=Vec::with_capacity(ts.len());for t in ts{let(a,b)=f.predict(t)?;m.push(a);v.push(b);}Ok((m,v))}
    fn likelihood(&self,kl:bool)->PyResult<f64>{self.inner.borrow().likelihood(kl)}
    fn identity(&self)->Vec<Vec<f64>>{Mat::eye(self.inner.borrow().h.n).rows()}
    fn matrices(&self,name:&str)->PyResult<Vec<Vec<Vec<f64>>>>{let f=self.inner.borrow();let a=match name{"_A"=>&f.a,"_Q"=>&f.q,"_m_p"=>&f.mp,"_P_p"=>&f.pp,"_m_f"=>&f.mf,"_P_f"=>&f.pf,"_m_s"=>&f.sm,"_P_s"=>&f.sp,_=>return Err(PyValueError::new_err("unknown state"))};Ok(a.rows())}
    fn dump(&self)->PyResult<Vec<u8>>{bincode::serialize(&*self.inner.borrow()).map_err(|e|PyValueError::new_err(e.to_string()))}
    fn restore(&self,data:Vec<u8>,kernel:PyRef<'_,NativeKernel>)->PyResult<()>{let mut f:FData=bincode::deserialize(&data).map_err(|e|PyValueError::new_err(e.to_string()))?;f.kernel=kernel.inner.clone();*self.inner.borrow_mut()=f;Ok(())}
}
