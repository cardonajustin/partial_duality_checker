include("dual_check.jl")
using Random, Serialization, Statistics, Plots, ProgressMeter, JacobiDavidson

# Random.seed!(0)
# CURAND.seed!(0)
# C = ConstraintFunction(2, T=ComplexF64)
# x = range(247.5, 247.6, 100)
# y = map(z-> C * eltype(C)(z), x)
# println(C * zero_est(C))
# display(plot(x, y))
# readline()
# 
depth = 5
dx = ComplexF32.([1e-3])
l_init = 10
params = collect(Iterators.product(dx, l_init))
errs = zeros(Number, 1000, depth + length(params[1]))

n = size(errs, 1)
pb = Progress(n)
for i in 1:n
	for p in params
		Random.seed!(1)
		CURAND.seed!(1)
		C = ConstraintFunction(2)
		Random.seed!()
		CURAND.seed!()
		z_init = eltype(C)(247.5949)# zero_est(C)
		# println(z_init)
		z, e = optimize(C, z_init=z_init, dx = p[1], l_init=p[2], iter=5, window=20)
		errs[i, :] = vcat(e[:, 2], collect(p))
		next!(pb)
	end
end
finish!(pb)
serialize("convergence.dat", errs)


# x2:	1 -> 247.5949
#		2 -> 435.703
#		3 -> 227.024
#		39 -> 442.7883
#
# x16:	0 -> 10403.075
#		1 ->
#		2 ->
#		3 ->
#		4 ->
#		5 ->
#		6 ->
#		7 ->
#		8 ->
#		9 ->

data = deserialize("convergence.dat")
println(mean(data, dims=1))
println(std(data, dims=1))
display(histogram(data[:, 5]))
readline()
