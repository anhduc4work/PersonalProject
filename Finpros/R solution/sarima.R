sarima=function(data,p,d,q,P=0,D=0,Q=0,S=-1){ 
  n=length(data)
  constant=1:n
  xmean=matrix(1,n,1)
  if (d>0)  
  fitit=arima(data, order=c(p,d,q), seasonal=list(order=c(P,D,Q), period=S),xreg=constant,include.mean=F)
  if (d<.00001)
  fitit=arima(data, order=c(p,d,q), seasonal=list(order=c(P,D,Q), period=S),xreg=xmean,include.mean=F)
  if (d+D>1)
  fitit=arima(data, order=c(p,d,q), seasonal=list(order=c(P,D,Q), period=S))
  if (S < 0) goof=20 else goof=3*S
  tsdiag(fitit,gof.lag=goof)
  k=length(fitit$coef)
  BIC=log(fitit$sigma2)+(k*log(n)/n)
  AICc=log(fitit$sigma2)+((n+k)/(n-k-2))
  AIC=log(fitit$sigma2)+((n+2*k)/n)
  list(fit=fitit, AIC=AIC, AICc=AICc, BIC=BIC)
}

