use crate::{kernel::{KRef,NativeKernel},matrix::{Mat,History}};
use pyo3::{exceptions::{PyRuntimeError,PyValueError,PyNotImplementedError},prelude::*};
use serde::{Deserialize,Serialize};
use std::{cell::RefCell,rc::Rc,collections::HashMap};

pub type FRef=Rc<RefCell<FData>>;
#[derive(Serialize,Deserialize)]
pub struct FData{
    pub kernel:KRef,pub batch:bool,pub fitted:bool,pub ts:Vec<f64>,pub pending:Vec<f64>,pub ms:Vec<f64>,pub vs:Vec<f64>,pub ns:Vec<f64>,pub xs:Vec<f64>,
    pub h:Mat,pub a:History,pub q:History,pub mp:History,pub pp:History,pub mf:History,pub pf:History,pub sm:History,pub sp:History,
    pub km:Mat,pub cov:Mat,pub chol:Mat,pub wi:Mat,pub wv:Mat,
}
impl FData{
    pub fn new(kernel:KRef,batch:bool)->PyResult<Self>{let h=if batch{Mat::zero(0,1)}else{kernel.borrow().h()?};let n=h.n;Ok(Self{kernel,batch,fitted:true,ts:vec![],pending:vec![],ms:vec![],vs:vec![],ns:vec![],xs:vec![],h,a:History::new(n,n),q:History::new(n,n),mp:History::new(n,1),pp:History::new(n,n),mf:History::new(n,1),pf:History::new(n,n),sm:History::new(n,1),sp:History::new(n,n),km:Mat::zero(0,0),cov:Mat::zero(0,0),chol:Mat::zero(0,0),wi:Mat::zero(0,0),wv:Mat::zero(0,1)})}
    pub fn add(&mut self,t:f64)->usize{let i=self.ts.len()+self.pending.len();self.pending.push(t);self.fitted=false;i}
    pub fn allocate(&mut self)->PyResult<()>{self.allocate_cached(&mut HashMap::new())}
    pub fn allocate_cached(&mut self,cache:&mut HashMap<(usize,u64),(Mat,Mat)>)->PyResult<()>{
        if self.pending.is_empty() && !self.batch{return Ok(());}
        let k=self.kernel.borrow();
        let stationary=fn_stationary(&k);
        for &t in &self.pending {
            self.ms.push(0.0);self.vs.push(k.cov(t,t));self.ns.push(0.0);self.xs.push(0.0);
            if !self.batch {let n=self.h.n;let p=k.matrix("state",t,t)?;self.mp.push(Mat::zero(n,1));self.mf.push(Mat::zero(n,1));self.sm.push(Mat::zero(n,1));self.pp.push(p.clone());self.pf.push(p.clone());self.sp.push(p);
                if let Some(&prev)=self.ts.last(){let key=(Rc::as_ptr(&self.kernel) as usize,(t-prev).to_bits());let(a,q)=if stationary {if let Some(step)=cache.get(&key){step.clone()}else{let step=(k.matrix("transition",prev,t)?,k.matrix("noise",prev,t)?);if cache.len()>=1024{cache.clear();}cache.insert(key,step.clone());step}}else{(k.matrix("transition",prev,t)?,k.matrix("noise",prev,t)?)};self.a.push(a);self.q.push(q);}
            } self.ts.push(t);
        }
        self.pending.clear();
        if self.batch {let n=self.ts.len();self.km=Mat::zero(n,n);for i in 0..n{for j in 0..n{self.km[(i,j)]=k.cov(self.ts[i],self.ts[j]);}}}
        Ok(())
    }
    pub fn fit(&mut self)->PyResult<()>{
        if !self.pending.is_empty(){return Err(PyRuntimeError::new_err("new data since last call to `allocate()`"));}
        let n=self.ts.len();if n==0{self.fitted=true;return Ok(());}
        if self.batch{
            let mut b=Mat::eye(n);let mut diag=Mat::zero(n,n);for i in 0..n{diag[(i,i)]=self.xs[i].sqrt();for j in 0..n{b[(i,j)]+=self.xs[i].sqrt()*self.xs[j].sqrt()*self.km[(i,j)];}}
            self.chol=b.cholesky()?;let mat=self.chol.lower_solve(&diag);self.wi=mat.transpose().mul(&mat);let wik=self.wi.mul(&self.km);self.cov=self.km.sub(&self.km.mul(&wik));let ns=Mat::from(n,1,&self.ns);let m=self.cov.mul(&ns);self.ms.copy_from_slice(&m.v);for i in 0..n{self.vs[i]=self.cov[(i,i)];}self.wv=ns.sub(&wik.mul(&ns));
        }else{
            let h=&self.h;let ht=h.transpose();let id=Mat::eye(h.n);
            for i in 0..n{
                if i>0{self.mp.set(i,self.a.get(i-1).mul(&self.mf.get(i-1)));self.pp.set(i,self.a.get(i-1).mul(&self.pf.get(i-1)).mul(&self.a.get(i-1).transpose()).add(&self.q.get(i-1)));}
                let gain=self.pp.get(i).mul(h).scale(1.0/(1.0+self.xs[i]*self.pp.get(i).quad(h)));self.mf.set(i,self.mp.get(i).add(&gain.scale(self.ns[i]-self.xs[i]*ht.mul(&self.mp.get(i)).v[0])));let z=id.sub(&gain.mul(&ht).scale(self.xs[i]));self.pf.set(i,z.mul(&self.pp.get(i)).mul(&z.transpose()).add(&gain.mul(&gain.transpose()).scale(self.xs[i])));
            }
            for i in (0..n).rev(){if i==n-1{self.sm.set(i,self.mf.get(i).clone());self.sp.set(i,self.pf.get(i).clone());}else{let g=self.pp.get(i+1).solve(&self.a.get(i).mul(&self.pf.get(i)))?.transpose();self.sm.set(i,self.mf.get(i).add(&g.mul(&self.sm.get(i+1).sub(&self.mp.get(i+1)))));self.sp.set(i,self.pf.get(i).add(&g.mul(&self.sp.get(i+1).sub(&self.pp.get(i+1))).mul(&g.transpose())));}self.ms[i]=ht.mul(&self.sm.get(i)).v[0];self.vs[i]=self.sp.get(i).quad(h);}
        }self.fitted=true;Ok(())
    }
    pub fn ready(&self)->PyResult<()>{if !self.fitted{Err(PyRuntimeError::new_err("new data since last call to `fit()`"))}else{Ok(())}}
    pub fn predict(&self,t:f64)->PyResult<(f64,f64)>{
        self.ready()?;let k=self.kernel.borrow();let n=self.ts.len();if n==0{return Ok((0.0,k.cov(t,t)));}
        if self.batch{let row=Mat::from(1,n,&self.ts.iter().map(|s|k.cov(t,*s)).collect::<Vec<_>>());return Ok((row.mul(&self.wv).v[0],k.cov(t,t)-row.mul(&self.wi).mul(&row.transpose()).v[0]));}
        let nxt=self.ts.partition_point(|s|*s<t);let h=&self.h;let ht=h.transpose();
        if nxt==n{let a=k.matrix("transition",self.ts[n-1],t)?;let q=k.matrix("noise",self.ts[n-1],t)?;return Ok((ht.mul(&a).mul(&self.sm.get(n-1)).v[0],a.mul(&self.sp.get(n-1)).mul(&a.transpose()).add(&q).quad(h)));}
        let(m,p)=if nxt==0{(Mat::zero(h.n,1),k.matrix("state",t,t)?)}else{let a=k.matrix("transition",self.ts[nxt-1],t)?;let q=k.matrix("noise",self.ts[nxt-1],t)?;(a.mul(&self.mf.get(nxt-1)),a.mul(&self.pf.get(nxt-1)).mul(&a.transpose()).add(&q))};
        let a=k.matrix("transition",t,self.ts[nxt])?;let g=self.pp.get(nxt).solve(&a.mul(&p))?.transpose();Ok((ht.mul(&m.add(&g.mul(&self.sm.get(nxt).sub(&self.mp.get(nxt))))).v[0],p.add(&g.mul(&self.sp.get(nxt).sub(&self.pp.get(nxt))).mul(&g.transpose())).quad(h)))
    }
    pub fn likelihood(&self,kl:bool)->PyResult<f64>{
        if kl && self.batch{return Err(PyNotImplementedError::new_err(""));}self.ready()?;let n=self.ts.len();if n==0{return Ok(0.0);}
        if self.batch{return Ok(-(0..n).map(|i|self.chol[(i,i)].ln()).sum::<f64>()+0.5*self.ns.iter().zip(&self.ms).map(|(a,b)|a*b).sum::<f64>());}
        let mut val=0.0;for i in 0..n{let m=self.h.transpose().mul(&self.mp.get(i)).v[0];let v=self.pp.get(i).quad(&self.h);let x=self.xs[i];let u=self.ns[i];val+=if kl{-0.5*((x*v+1.0).ln()+x*(m*m-self.ms[i]*self.ms[i]-self.vs[i])-2.0*u*(m-self.ms[i])-(x*m-u).powi(2)/(1.0/v+x))}else{-0.5*((x*v+1.0).ln()+(-u*u*v-2.0*u*m+x*m*m)/(x*v+1.0))};}Ok(val)
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
    fn add_sample(&self,t:f64)->usize{self.inner.borrow_mut().add(t)}
    fn allocate(&self)->PyResult<()>{self.inner.borrow_mut().allocate()}
    fn fit(&self)->PyResult<()>{self.inner.borrow_mut().fit()}
    fn ready(&self)->PyResult<()>{self.inner.borrow().ready()}
    fn is_allocated(&self)->bool{self.inner.borrow().pending.is_empty()}
    fn is_fitted(&self)->bool{self.inner.borrow().fitted}
    fn set_fitted(&self,v:bool){self.inner.borrow_mut().fitted=v;}
    fn get_array(&self,name:&str)->PyResult<Vec<f64>>{Ok(self.inner.borrow().array(name)?.clone())}
    fn set_array(&self,name:&str,v:Vec<f64>)->PyResult<()>{let mut f=self.inner.borrow_mut();if name!="ts_new" && v.len()!=f.ts.len(){return Err(PyValueError::new_err("array length must match allocated samples"));}match name{"ts"=>f.ts=v,"ts_new"=>f.pending=v,"ms"=>f.ms=v,"vs"=>f.vs=v,"ns"=>f.ns=v,"xs"=>f.xs=v,_=>return Err(PyValueError::new_err("unknown array"))}Ok(())}
    fn predict(&self,ts:Vec<f64>)->PyResult<(Vec<f64>,Vec<f64>)>{let f=self.inner.borrow();f.ready()?;let mut m=Vec::with_capacity(ts.len());let mut v=Vec::with_capacity(ts.len());for t in ts{let(a,b)=f.predict(t)?;m.push(a);v.push(b);}Ok((m,v))}
    fn likelihood(&self,kl:bool)->PyResult<f64>{self.inner.borrow().likelihood(kl)}
    fn matrices(&self,name:&str)->PyResult<Vec<Vec<Vec<f64>>>>{let f=self.inner.borrow();let a=match name{"_A"=>&f.a,"_Q"=>&f.q,"_m_p"=>&f.mp,"_P_p"=>&f.pp,"_m_f"=>&f.mf,"_P_f"=>&f.pf,"_m_s"=>&f.sm,"_P_s"=>&f.sp,_=>return Err(PyValueError::new_err("unknown state"))};Ok(a.rows())}
    fn dump(&self)->PyResult<Vec<u8>>{bincode::serialize(&*self.inner.borrow()).map_err(|e|PyValueError::new_err(e.to_string()))}
    fn restore(&self,data:Vec<u8>,kernel:PyRef<'_,NativeKernel>)->PyResult<()>{let mut f:FData=bincode::deserialize(&data).map_err(|e|PyValueError::new_err(e.to_string()))?;f.kernel=kernel.inner.clone();*self.inner.borrow_mut()=f;Ok(())}
}
