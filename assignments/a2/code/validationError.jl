using Printf

# Load X and y variable
using JLD
X = load("citiesSmall.jld","X")
y = load("citiesSmall.jld","y")
n = size(X,1)

X_train = X[1:n/2,:]
y_train = y[1:n/2]
X_validation = X[n/2:end,:]
y_validation = y[n/2:end]

model = decisionTree_infoGain(X,y,depth)

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