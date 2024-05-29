beer<-scan("beer.txt")
plot.ts(beer, type="b", main="Beer Production in Australia")

# Check for season
par(mfrow=c(4,1))
plot.ts(diff(beer,1), main="First Difference (Non-Seasonal)")
plot.ts(diff(beer,2), main="2nd Difference (Non-Seasonal)")
plot.ts(diff(beer,5), main="3rd Difference (Non-Seasonal)")
plot.ts(diff(beer,4), main="Fourth Difference (Seasonal)")
# diff4 disappear season -> chose 4

#Plot new
plot.ts(diff(diff(beer),4), main="Time Series Plot with both non-seasonal and seasonal diff")


# Chose order
par(mfrow=c(2,1))
acf(diff(diff(beer),4),16, main="ACF with both differencing")
pacf(diff(diff(beer),4),16, main="PACF with both differencing")
# ACF peak at 4, PACF not significant -> (0,1,1)_12
# within season, p=2,0  q =1,3,0,  -> 6 models


# Model diagnostics
source("sarima.R")
#fit1<-sarima(beer,2,1,0,0,1,2,4) ##ok. barely sig acf at lag 9
fit2<-sarima(beer,0,1,1,0,1,2,4) ##pvalue not good
fit3<-sarima(beer,1,1,1,0,1,2,4) ##ok
fit4<-sarima(beer,2,1,1,0,1,2,4) #ok
fit5<-sarima(beer,2,1,3,0,1,2,4) #ok
fit6<-sarima(beer,0,1,3,0,1,2,4) #ok
fit7<-sarima(beer,0,1,0,0,1,2,4) #no good


# Model selection
fit3
fit4
# AIC and AICc prefer model 3, BIC prefer model 4

# Forecast
fit_best = arima(beer, order=c(1,1,1), seasonal=list(order=c(0,1,2), period=4))
predict(fit_best,12)
