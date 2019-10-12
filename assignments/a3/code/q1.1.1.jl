# Load data
using JLD
X = load("clusterData2.jld","X")
k = 5
include("kMeans.jl")
model = kMeans(X,k)
y = model.predict(X)

minModel = model
minError = kMeansError2(X,y,model.W)

for i in 1:100
    model = kMeans(X,k)
    y = model.predict(X)
    err = kMeansError2(X,y,model.W)
    if err < minError
        global minError = err
        global minModel = model
    end
end

include("clustering2Dplot.jl")
y = minModel.predict(X)
@show kMeansError2(X,y,minModel.W)
clustering2Dplot(X,y,minModel.W)
