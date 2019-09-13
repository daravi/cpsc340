include("misc.jl") # Includes "mode" function

# First let's define a "type",
# specifying the functions/varaibles we want the stump to have
mutable struct StumpModel
	predict # Function that makes predictions
	split # Function that splits data
	baseSplit # Set this to one stump doesn't split
end

function decisionStumpEquality(X,y)
	# Fits a decision stump based on equality after rounding to nearest integer

	# Get the size of the data matrix
	(n,d) = size(X)

	# Round all the data to the nearest integer
	X = round.(X)

	# Initialize the "best rule" with the baseline rule (no split)
	y_mode = mode(y)
	minError = sum(y .!= y_mode);
	splitVariable = [];
	splitValue = [];
	splitYes = y_mode;
	splitNo = [];

	# Search for the best rule
	# (Uses O(n^2d) approach to keep code simple)
	yhat = zeros(n)
	for j in 1:d
		# Try unique values of column as split values
		for val in unique(X[:,j])

			# Test whether each object satisfies equality
			yes = X[:,j] .== val

			# Find correct label on both sides of split
			y_yes = mode(y[yes])
			y_no = mode(y[.!yes])

			# Make predictions
			yhat[yes] .= y_yes
			yhat[.!yes] .= y_no

			# Compute error
			trainError = sum(yhat .!= y)

			# Update best rule
			if trainError < minError
				minError = trainError
				splitVariable = j
				splitValue = val
				splitYes = y_yes
				splitNo = y_no
			end
		end
	end

	# Now that we have the best rule,
	# let's build our splitting function
	function split(Xhat)
		(t,d) = size(Xhat)
		Xhat = round.(Xhat)
		if isempty(splitVariable)
			return fill(true,t)
		else
			return (Xhat[:,splitVariable] .== splitValue)
		end
	end

	# Now that we have the best rule,
	# let's build our predict function
	function predict(Xhat)
		(t,d) = size(Xhat)
		yes = split(Xhat)
		yhat = fill(splitYes,t)
		if any(.!yes)
			yhat[.!yes] .= splitNo
		end
		return yhat
	end

	return StumpModel(predict,split,isempty(splitNo))
end


function decisionStump(X,y)
	# Fits a decision stump based on inequality

	# Get the size of the data matrix
	(n,d) = size(X)

	# Initialize the "best rule" with the baseline rule (no split)
	y_mode = mode(y)
	minError = sum(y .!= y_mode);
	splitVariable = [];
	splitValue = [];
	splitLE = y_mode; # predicted label for less than or equal to the split value
	splitGT = [];     # predicted label for greater than the split value

	# Search for the best rule
	# (Uses O(n^2d) approach to keep code simple)
	yhat = zeros(n)
	for j in 1:d
		# Try unique values of column as split values
		for val in unique(X[:,j])

			# Test whether each object satisfies inequality
			lessThanOrEqual = X[:,j] .<= val

			# Find correct label on both sides of split
			y_le = mode(y[lessThanOrEqual])
			y_gt = mode(y[.!lessThanOrEqual])

			# Make predictions
			yhat[lessThanOrEqual] .= y_le
			yhat[.!lessThanOrEqual] .= y_gt

			# Compute error
			trainError = sum(yhat .!= y)

			# Update best rule
			if trainError < minError
				minError = trainError
				splitVariable = j
				splitValue = val
				splitLE = y_le
				splitGT = y_gt
			end
		end
	end

	@show splitVariable, splitValue, splitLE, splitGT

	# Now that we have the best rule,
	# let's build our splitting function
	function split(Xhat)
		(t,d) = size(Xhat)
		if isempty(splitVariable)
			return fill(true,t)
		else
			return (Xhat[:,splitVariable] .<= splitValue)
		end
	end

	# Now that we have the best rule,
	# let's build our predict function
	function predict(Xhat)
		(t,d) = size(Xhat)
		lessThanOrEqual = split(Xhat)
		yhat = fill(splitLE,t)
		if any(.!lessThanOrEqual)
			yhat[.!lessThanOrEqual] .= splitGT
		end
		return yhat
	end

	return StumpModel(predict,split,isempty(splitGT))
end
