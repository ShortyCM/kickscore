use crate::matrix::Mat;
use pyo3::{exceptions::{PyNotImplementedError,PyValueError},prelude::*};
use std::{f64::consts::PI,sync::OnceLock};

pub fn eigen(mut a:Mat)->PyResult<(Vec<f64>,Mat)>{
    if a.v.iter().any(|x|!x.is_finite()){return Err(crate::matrix::linear_error("Eigenvalues did not converge"));}
    let n=a.n;let mut v=Mat::eye(n);
    let mut converged=n<2;
    for _ in 0..100*n*n {
        let(mut p,mut q,mut largest)=(0,0,0.0);
        for i in 0..n {for j in i+1..n {if a[(i,j)].abs()>largest {p=i;q=j;largest=a[(i,j)].abs();}}}
        if largest==0.0 {converged=true;break;}
        let scale=(0..n).map(|i|a[(i,i)].abs()).fold(0.0,f64::max);
        if largest<=f64::EPSILON*scale {converged=true;break;}
        let tau=(a[(q,q)]-a[(p,p)])/(2.0*a[(p,q)]);let t=if tau>=0.0{1.0/(tau+tau.hypot(1.0))}else{-1.0/(-tau+tau.hypot(1.0))};let c=1.0/(1.0+t*t).sqrt();let s=t*c;
        let ap=a[(p,p)];let aq=a[(q,q)];let off=a[(p,q)];a[(p,p)]=ap-t*off;a[(q,q)]=aq+t*off;a[(p,q)]=0.0;a[(q,p)]=0.0;
        for i in 0..n {if i!=p && i!=q {let x=a[(i,p)];let y=a[(i,q)];a[(i,p)]=c*x-s*y;a[(p,i)]=a[(i,p)];a[(i,q)]=s*x+c*y;a[(q,i)]=a[(i,q)];}let x=v[(i,p)];let y=v[(i,q)];v[(i,p)]=c*x-s*y;v[(i,q)]=s*x+c*y;}
    }
    if !converged{return Err(crate::matrix::linear_error("Eigenvalues did not converge"));}
    Ok(((0..n).map(|i|a[(i,i)]).collect(),v))
}
fn hermite(x:f64)->(f64,f64) {
    let mut previous=1.0;
    let mut current=x;
    for i in 2..=30 {
        let next=(x*current-((i-1) as f64).sqrt()*previous)/(i as f64).sqrt();
        previous=current;
        current=next;
    }
    (current,30.0_f64.sqrt()*previous)
}

fn quadrature()->PyResult<&'static [(f64,f64);30]> {
    static DATA:OnceLock<[(f64,f64);30]>=OnceLock::new();
    if let Some(data)=DATA.get(){return Ok(data);}
    let mut matrix=Mat::zero(30,30);
    for i in 1..30 {matrix[(i-1,i)]=(i as f64).sqrt();matrix[(i,i-1)]=(i as f64).sqrt();}
    let(mut nodes,_)=eigen(matrix)?;
    nodes.sort_by(f64::total_cmp);
    let mut out=[(0.0,0.0);30];
    for i in 0..15 {
        let mut x=(nodes[29-i]-nodes[i])/2.0;
        let mut converged=false;
        for _ in 0..20 {
            let(value,derivative)=hermite(x);
            let change=value/derivative;
            x-=change;
            if change.abs()<=2.0*f64::EPSILON*x.abs().max(1.0){converged=true;break;}
        }
        if !converged{return Err(crate::matrix::linear_error("Hermite roots did not converge"));}
        let derivative=hermite(x).1;
        let weight=1.0/(derivative*derivative);
        out[i]=(-x,weight);
        out[29-i]=(x,weight);
    }
    let sum: f64=out.iter().map(|(_,w)|w).sum();
    for (_,weight) in &mut out {*weight/=sum;}
    let _=DATA.set(out);
    Ok(DATA.get().expect("quadrature initialized"))
}
#[pyfunction]
pub fn normpdf(x:f64)->f64{(-x*x/2.0).exp()/(2.0*PI).sqrt()}
#[pyfunction]
pub fn normcdf(x:f64)->f64{libm::erfc(-x/2.0_f64.sqrt())/2.0}
#[pyfunction]
pub fn logphi(z:f64)->(f64,f64){
    let sqrt2pi=(2.0*PI).sqrt();
    if z*z<0.0492 {let cs=[0.00048204,-0.00142906,0.0013200243174,0.0009461589032,-0.0045563339802,0.00556964649138,0.00125993961762116,-0.01621575378835404,0.02629651521057465,-0.001829764677455021,2.0*(1.0-PI/3.0),(4.0-PI)/3.0,1.0,1.0];let mut val=0.0;for c in cs{val=-z/sqrt2pi*(c+val);}let r=-2.0*val-2.0_f64.ln();(r,(-z*z/2.0-r).exp()/sqrt2pi)}
    else if z< -11.3137 {let mut num=0.5641895835477550741;for r in [1.2753666447299659525,5.019049726784267463450,6.1602098531096305441,7.409740605964741794425,2.9788656263939928886]{num=-z*num/2.0_f64.sqrt()+r;}let mut den=1.0;for q in [2.260528520767326969592,9.3960340162350541504,12.048951927855129036034,17.081440747466004316,9.608965327192787870698,3.3690752069827527677]{den=-z*den/2.0_f64.sqrt()+q;}((num/(2.0*den)).ln()-z*z/2.0,(den/num).abs()*(2.0/PI).sqrt())}
    else{let r=normcdf(z).ln();(r,(-z*z/2.0-r).exp()/sqrt2pi)}
}
#[pyfunction]
pub fn log_factorial(k:usize)->f64 {
    static CACHE:OnceLock<[f64;500]>=OnceLock::new();
    let cache=CACHE.get_or_init(||{let mut out=[0.0;500];for i in 1..500{out[i]=out[i-1]+(i as f64).ln();}out});
    if k<500{return cache[k];}
    let mut value=cache[499];
    for i in 500..=k{value+=(i as f64).ln();}
    value
}
#[pyfunction]
pub fn logsumexp(xs:Vec<f64>)->PyResult<f64>{
    if xs.is_empty(){return Err(PyValueError::new_err("zero-size array to reduction operation maximum which has no identity"));}
    let a=xs.iter().copied().fold(f64::NEG_INFINITY,f64::max);
    Ok(a+xs.iter().map(|x|(x-a).exp()).sum::<f64>().ln())
}
#[pyfunction]
pub fn logsumexp2(xs:Vec<f64>,bs:Vec<f64>)->PyResult<f64>{
    if xs.is_empty(){return Err(PyValueError::new_err("zero-size array to reduction operation maximum which has no identity"));}
    if bs.len()!=xs.len()&&bs.len()!=1&&xs.len()!=1{return Err(PyValueError::new_err("operands could not be broadcast together"));}
    let a=xs.iter().copied().fold(f64::NEG_INFINITY,f64::max);
    let n=if bs.is_empty(){0}else{xs.len().max(bs.len())};
    Ok(a+(0..n).map(|i|bs[if bs.len()==1{0}else{i}]*(xs[if xs.len()==1{0}else{i}]-a).exp()).sum::<f64>().ln())
}
pub fn probit(m:f64,v:f64)->(f64,f64,f64){let z=m/(1.0+v).sqrt();let(l,a)=logphi(z);(l,a/(1.0+v).sqrt(),-a*(z+a)/(1.0+v))}
pub fn probit_tie(m:f64,v:f64,margin:f64)->PyResult<(f64,f64,f64)>{if margin<=0.0{return Err(PyValueError::new_err("tie margin must be positive"));}let d=(1.0+v).sqrt();let z1=(-m.abs()+margin)/d;let z2=(-m.abs()-margin)/d;let l1=logphi(z1).0;let l=l1+(-(logphi(z2).0-l1).exp_m1()).ln();let a=(-0.5*z1*z1-(2.0*PI).sqrt().ln()-l).exp();let b=(-0.5*z2*z2-(2.0*PI).sqrt().ln()-l).exp();let mut first=(a-b)/d;let second=(-z1*a+z2*b)/(1.0+v)-first*first;if m>0.0{first=-first;}Ok((l,first,second))}
pub fn logit(m:f64,v:f64)->(f64,f64,f64){
    let exponent=-10.0*(m.abs()-196.0/200.0*v-4.0);
    let(lam,l2,d2)=if exponent<500.0{let lam=1.0/(1.0+exponent.exp());let l=(v/2.0-m.abs()).min(-0.1);if m>0.0{(lam,(-l.exp()).ln_1p(),0.0)}else{(lam,l,1.0)}}else{(0.0,0.0,0.0)};
    if lam==1.0{return(l2,d2,0.0);}
    let cs=[1.146480988574439e02,-1.508871030070582e03,2.676085036831241e03,-1.356294962039222e03,7.543285642111850e01];let ls=[0.44,0.41,0.40,0.39,0.36].map(|x|x*2.0_f64.sqrt());let vals=ls.map(|x|probit(x*m,x*x*v));let max=vals.iter().map(|a|a.0).fold(f64::NEG_INFINITY,f64::max);let(mut den,mut d,mut dd)=(0.0,0.0,0.0);
    for i in 0..5{let w=(vals[i].0-max).exp();den+=w*cs[i];d+=(w*vals[i].1)*(cs[i]*ls[i]);dd+=(w*(vals[i].1*vals[i].1+vals[i].2))*(cs[i]*ls[i]*ls[i]);}d/=den;dd=dd/den-d*d;((1.0-lam)*(max+den.ln())+lam*l2,(1.0-lam)*d+lam*d2,(1.0-lam)*dd)
}
fn log_iv(order:f64,z:f64)->f64{
    if z==0.0{return if order==0.0{0.0}else{f64::NEG_INFINITY};}
    if z.is_nan()||order.is_nan(){return f64::NAN;}
    if z==f64::INFINITY{return f64::INFINITY;}
    let loghalf=if z< f64::MIN_POSITIVE{z.ln()-2.0_f64.ln()}else{(z/2.0).ln()};
    let mut term=order*loghalf-libm::lgamma(order+1.0);let mut sum=term;let logz=2.0*loghalf;
    let mut k:f64=1.0;loop {term+=logz-k.ln()-(order+k).ln();let max=sum.max(term);let next=max+((sum-max).exp()+(term-max).exp()).ln();if next>f64::MAX.ln(){return f64::INFINITY;}if next==sum{return sum;}if next.is_nan(){return f64::NAN;}sum=next;k+=1.0;}
}
#[pyfunction]
pub fn iv(order:f64,z:f64)->f64{log_iv(order.abs(),z).exp()}
pub fn ll(kind:u8,x:f64,p:f64,q:f64)->f64{
    let sigmoid=|z:f64|if z>0.0{-(-z).exp().ln_1p()}else{z-z.exp().ln_1p()};
    match kind{0=>logphi(x-p).0,1=>sigmoid(x-p),2=>{let x=-x.abs();let z=logphi(x+p).0;let a=logphi(x-p).0-z;z+if a> -0.693{(-a.exp_m1()).ln()}else{(-a.exp()).ln_1p()}},3=>sigmoid(x-p)+sigmoid(-x-p)+(2.0*p).exp_m1().ln(),4=>-0.5*((2.0*PI*q).ln()+(p-x).powi(2)/q),5=>x*p-log_factorial(p as usize)-x.exp(),6=>-((x+q).exp()+(-x+q).exp())+x*p+iv(p.abs(),2.0*q.exp()).ln(),_=>f64::NAN}
}
#[pyfunction]
pub fn moments(kind:u8,m:f64,v:f64,p:f64,q:f64,kl:bool)->PyResult<(f64,f64,f64)>{
    if kind>6{return Err(PyNotImplementedError::new_err("unknown observation type"));}
    if !kl&&((kind==0&&v== -1.0)||(kind==2&&p>0.0&&v== -1.0)||(kind==4&&q+v==0.0)){return Err(pyo3::exceptions::PyZeroDivisionError::new_err("division by zero"));}
    if !kl{match kind{0=>return Ok(probit(m-p,v)),1=>return Ok(logit(m-p,v)),2=>return probit_tie(m,v,p),4=>return Ok((-0.5*((2.0*PI*(q+v)).ln()+(p-m).powi(2)/(q+v)),(p-m)/(q+v),-1.0/(q+v))),_=>{}}}
    if kl && kind==4{return Err(PyNotImplementedError::new_err(""));}
    if v==0.0{return Err(pyo3::exceptions::PyZeroDivisionError::new_err("division by zero"));}
    let constant=match kind {5=>-log_factorial(p as usize),6=>iv(p.abs(),2.0*q.exp()).ln(),_=>0.0};
    let eval=|x:f64|match kind{5=>x*p+constant-x.exp(),6=>-((x+q).exp()+(-x+q).exp())+x*p+constant,_=>ll(kind,x,p,q)};
    let std=v.sqrt();let data=quadrature()?;let(mut value,mut first,mut second)=(0.0,0.0,0.0);
    if kl {for &(x,w) in data{let val=w*eval(std*x+m);value+=val;first+=x/std*val;second+=(x*x-1.0)/(2.0*v)*val;}}
    else{let mut vals=[0.0;30];let mut max=f64::NEG_INFINITY;for(i,&(x,w))in data.iter().enumerate(){vals[i]=w.ln()+eval(std*x+m);max=max.max(vals[i]);}value=max+vals.iter().map(|x|(x-max).exp()).sum::<f64>().ln();for(i,&(x,_))in data.iter().enumerate(){let w=(vals[i]-value).exp();first+=w*x/std;second+=w*(x*x-1.0)/v;}second-=first*first;}
    Ok((value,first,second))
}
#[pyfunction]
pub fn loglike(kind:u8,x:f64,p:f64,q:f64)->f64{ll(kind,x,p,q)}
#[pyfunction]
pub fn probability(kind:u8,m:f64,v:f64,p:f64,q:f64)->PyResult<f64>{if kind==4&&q+v<0.0{return Err(PyValueError::new_err("math domain error"));}Ok(match kind{2 if p==0.0=>0.0,3=>1.0-logit(m-p,v).0.exp()-logit(-m-p,v).0.exp(),4=>normcdf((m-p)/(q+v).sqrt()),_=>moments(kind,m,v,p,q,false)?.0.exp()})}
#[pyfunction]
pub fn distances(a:Vec<f64>,b:Vec<f64>)->Vec<Vec<f64>>{a.iter().map(|x|b.iter().map(|y|(x-y).abs()).collect()).collect()}
