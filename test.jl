include("dual_check.jl")
using Random, Serialization, Statistics, Plots, ProgressMeter

Random.seed!(3)
CURAND.seed!(3)

C = ConstraintFunction(16)
x = range(400, 500, 100)
y = map(z-> C * ComplexF32(z), x)
display(plot(x, y))
readline()

# depth = 5
# dx = ComplexF32.([1e-3])
# l_init = 10
# params = collect(Iterators.product(dx, l_init))
# errs = zeros(Number, 10000, depth + length(params[1]))
# 
# n = size(errs, 1)
# pb = Progress(n)
# for i in 1:n
# 	for p in params
# 		Random.seed!(2)
# 		CURAND.seed!(2)
# 		C = ConstraintFunction(2)
# 		Random.seed!()
# 		CURAND.seed!()
# 		z, e = optimize(C, z_init=ComplexF32(435.703 - 0.5 + rand()), dx = p[1], l_init=p[2], iter=5)
# 		errs[i, :] = vcat(e[:, 2], collect(p))
# 		next!(pb)
# 	end
# end
# finish!(pb)
# 
# serialize("convergence.dat", errs)
# x:	1 -> 247.5949
#		2 -> 435.703
#		3 -> 227.024
#		39 -> 442.7883

# data = deserialize("convergence.dat")
# println(mean(data, dims=1))
# println(std(data, dims=1))
# display(histogram(data[:, 5]))
# readline()
