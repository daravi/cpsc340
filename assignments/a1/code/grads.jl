using Test

### An example to get you started:

# Function
func0(x) = sum(x.^2)

# Gradient
grad0(x) = 2x

### Function 1

function func1(x)
	f = 0;
	for x_i in x
		f += x_i^3;
	end
	return f
end

function grad1(x)
	n = length(x);
	g = zeros(n);
	for i in 1:n
		g[i] = 3 * (x[i]^2)
	end
	return g
end

### Function 2
func2(x) = prod(x)

function grad2(x)
	n = length(x);
	g = zeros(n);
	# Put gradient code here
	return g
end

### Function 3
func3(x) = -sum(log.(1 .+ exp.(-x)))

function grad3(x)
	# Put gradient code here	
end

### A function to approximate the derivative numerically
function numGrad(func,x)
	n = length(x);
	delta = 1e-6;
	g = zeros(n);
	fx = func(x);
	for i = 1:n
		e_i = zeros(n);
		e_i[i] = 1;
		g[i] = (func(x[:] + delta*e_i) - fx)/delta;
	end
	return g
end

@test grad0([10, 2]) ≈ numGrad(func0, [10 2]) atol=0.0001
@test grad1([10, 2]) ≈ numGrad(func1, [10 2]) atol=0.0001
# @test grad2([10, 2]) ≈ numGrad(func2, [10 2]) atol=0.0001
# @test grad3([10, 2]) ≈ numGrad(func3, [10 2]) atol=0.0001
