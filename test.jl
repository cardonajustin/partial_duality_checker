include("dual_check.jl")
using Random, Serialization, Statistics, Plots, ProgressMeter

# Random.seed!(0)
# CURAND.seed!(0)
# C = ConstraintFunction(2)
# z_g = 247.5949
# x = range(247.5, 247.61, 100)
# y = map(z-> C * z, x)
# display(plot(real.(x), y))
# readline()
# c = norm(maximum(C.G))
# perturb(C, 5e-2 * c)
# y = map(z-> C * z, x)
# display(plot(real.(x), y))
# readline()
# #  
depth = 5
dx = ComplexF32.([1e-4, 1e-3, 1e-2])
l_init = 10:12
params = collect(Iterators.product(dx, l_init))
n = 100
m = length(params)
errs = zeros(Number, n * m, depth + length(params[1]))

pb = Progress(n*m)


n_C = 12
# Cs = [ConstraintFunction(n_C) for i in 1:n]
for i in 1:n
	C = ConstraintFunction(n_C)
	for j in 1:m
		next!(pb)
		while true
			try
				z_init = zero_est(C) + eltype(C)(3)
				z, e = optimize(C, z_init=z_init, dx = params[j][1], l_init=params[j][2], iter=5, window=20)
				errs[((i - 1) * m) + j, :] = vcat(e[:, 2], collect(params[j]))
				break
			catch
				C = ConstraintFunction(n_C)
				continue
			end
		end
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
println(size(data))
println(mean(data, dims=1))
println(std(data, dims=1))
#display(histogram(data[:, 5]))
#readline()
