include("decisionTree_infoGain.jl")
using JLD

maxDepth = 15

X = load("citiesSmall.jld","X")
y = load("citiesSmall.jld","y")
n = size(X,1)

Xtest = load("citiesSmall.jld","Xtest")
ytest = load("citiesSmall.jld","ytest")
t = size(Xtest,1)

trainErrors = zeros(maxDepth)
testErrors = zeros(maxDepth)
for depth in 1:maxDepth
    model = decisionTree_infoGain(X,y,depth)

    yhat = model.predict(X)
    trainError = sum(yhat .!= y)/n
    trainErrors[depth] = trainError
    
    yhat = model.predict(Xtest)
    testError = sum(yhat .!= ytest)/t
    testErrors[depth] = testError
end

using PyPlot
figure()
plot(1:maxDepth, testErrors,"b+")
plot(1:maxDepth, trainErrors,"b+")
# plot(X[y.==2,1],X[y.==2,2],"ro")