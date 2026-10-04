use pyo3::{exceptions::PyValueError,prelude::*};

pub struct RandomState { keys: Vec<u32>, position: usize, cached: Option<f64> }
impl RandomState {
    pub fn load(py: Python<'_>) -> PyResult<Self> {
        let state=py.import("numpy.random")?.call_method0("get_state")?;
        let name:String=state.get_item(0)?.extract()?;
        if name!="MT19937" {return Err(PyValueError::new_err("unsupported NumPy RandomState generator"));}
        let keys:Vec<u32>=state.get_item(1)?.call_method0("tolist")?.extract()?;
        let position=state.get_item(2)?.extract()?;
        let has:bool=state.get_item(3)?.is_truthy()?;
        let cached=if has{Some(state.get_item(4)?.extract()?)}else{None};
        Ok(Self{keys,position,cached})
    }
    pub fn save(&self,py:Python<'_>)->PyResult<()> {
        py.import("numpy.random")?.call_method1("set_state",(("MT19937",self.keys.clone(),self.position,usize::from(self.cached.is_some()),self.cached.unwrap_or(0.0)),))?;
        Ok(())
    }
    fn next(&mut self)->u32 {
        if self.position>=624 {
            for i in 0..624 {let y=(self.keys[i]&0x80000000)|(self.keys[(i+1)%624]&0x7fffffff);self.keys[i]=self.keys[(i+397)%624]^(y>>1)^if y&1!=0{0x9908b0df}else{0};}
            self.position=0;
        }
        let mut y=self.keys[self.position];self.position+=1;
        y^=y>>11;y^=(y<<7)&0x9d2c5680;y^=(y<<15)&0xefc60000;y^=y>>18;y
    }
    fn uniform(&mut self)->f64 {let a=self.next()>>5;let b=self.next()>>6;((a as f64)*67108864.0+b as f64)/9007199254740992.0}
    pub fn normal(&mut self)->f64 {
        if let Some(value)=self.cached.take(){return value;}
        loop {let x=2.0*self.uniform()-1.0;let y=2.0*self.uniform()-1.0;let r=x*x+y*y;if r>=1.0||r==0.0{continue;}let factor=(-2.0*r.ln()/r).sqrt();self.cached=Some(factor*x);return factor*y;}
    }
}
