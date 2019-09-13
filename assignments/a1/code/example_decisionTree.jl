# Load X and y variable
using JLD
using Printf

X = load("citiesSmall.jld","X")
y = load("citiesSmall.jld","y")
n = size(X,1)

# Fit a decision tree and compute error
include("decisionTree.jl")
depth = 2
model = decisionTree(X,y,depth)

# Evaluate training error
yhat = model.predict(X)
trainError = sum(yhat .!= y)/n
@printf("Error with depth-%d decision tree: %.3f\n",depth,trainError)

# Plot classifier
include("plot2Dclassifier.jl")
plot2Dclassifier(X,y,model)

function predict(Xhat)
	(t,d) = size(Xhat)
	yhat = zeros(t)
	for i in 1:t
		if Xhat[i,2] <= 37.669007
			if Xhat[i,1] <= -115.577574
				yhat[i] = 1
			else
				yhat[i] = 2
			end
		else
			if Xhat[i,1] <= -96.090109
				yhat[i] = 2
			else
				yhat[i] = 1
			end
		end
	end
	return yhat
end

yhat = predict(X)
trainError = sum(yhat .!= y)/n
@printf("Error with depth-%d decision tree: %.3f\n",depth,trainError)