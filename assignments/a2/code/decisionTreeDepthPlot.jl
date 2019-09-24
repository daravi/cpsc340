using Printf

# Load X and y variable
using JLD
X = load("citiesSmall.jld","X")
y = load("citiesSmall.jld","y")
n = size(X,1)
Xtest = load("citiesSmall.jld","Xtest")
ytest = load("citiesSmall.jld","ytest")
t = size(Xtest,1)

include("decisionTree_infoGain.jl")

function decisionTreeErrors(maxDepth)
    trainErrors = zeros(maxDepth)
    testErrors = zeros(maxDepth)
    for depth in 1:maxDepth
        model = decisionTree_infoGain(X,y,depth)
        yhat = model.predict(X)
        trainErrors[depth] = sum(yhat .!= y)/n
        yhat_test = model.predict(Xtest)
        testErrors[depth] = sum(yhat_test .!= ytest)/t
    end
    return trainErrors, testErrors
end

trainErrors, testErrors = decisionTreeErrors(15)