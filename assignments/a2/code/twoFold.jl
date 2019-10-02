include("decisionTree_infoGain.jl")
using JLD

maxDepth = 15

X = load("citiesSmall.jld","X")
y = load("citiesSmall.jld","y")
n = size(X,1)

X1 = X[1:n÷2,:]
y1 = y[1:n÷2]
n1 = n÷2

X2 = X[n÷2+1:end,:]
y2 = y[n÷2+1:end]
n2 = n÷2

# trainErrors = zeros(maxDepth)
validationErrors = zeros(maxDepth)
for depth in 1:maxDepth
    model = decisionTree_infoGain(X2,y2,depth)

    # yhat = model.predict(X)
    # trainError = sum(yhat .!= y)/n
    # trainErrors[depth] = trainError
    
    yhat = model.predict(X1)
    validationError = sum(yhat .!= y1)/n1
    validationErrors[depth] = validationError
end

# using PyPlot
# figure()
# plot(1:maxDepth, testErrors,"b+")
# plot(1:maxDepth, trainErrors,"b+")
# plot(X[y.==2,1],X[y.==2,2],"ro")