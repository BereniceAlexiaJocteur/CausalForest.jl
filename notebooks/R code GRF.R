setwd("/Users/jocteur/.julia/dev/CausalForest/notebooks")

#data = read.csv("sinus_causal_3.csv", dec=".")
#data = read.csv("sinus_causal_4.csv", dec=".")
#data = read.csv("sinus_causal_5.csv", dec=".")
data = read.csv("sinus_causal_6.csv", dec=".")


library("grf")
library("Metrics")

X = data[,1:10]
W = data[,11]
Y = data[,12]

cf = causal_forest(X, Y, W, num.trees = 500)
errors_1 = rep(0, 100)
for (i in 1:100){
  set.seed(i)
  Xtest = matrix(runif(10000,0,10),nrow=1000)
  pred = predict(cf, Xtest)$predictions
  true_effect = sin(Xtest[, 1])
  errors_1[i] = rmse(true_effect, pred)
}
mean(errors_1)
var(errors_1)

freq = split_frequencies(cf,20)
sum(freq[,1])/sum(freq)


set.seed(1)
Xtest = matrix(runif(10000,0,10),nrow=1000)
pred = predict(cf, Xtest)$predictions
true_effect = sin(Xtest[, 1])
idx <- order(Xtest[, 1])
plot(Xtest[idx, 1], pred[idx], type='l', col='red')
lines(Xtest[idx, 1], true_effect[idx], col="blue")
