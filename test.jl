include("dual_check.jl")
using Plots, Random

Random.seed!(1)
CURAND.seed!(1)
# C = ConstraintFunction(2)

# x = range(247.592, 247.597, 100)
# y = map(z-> C * ComplexF32(z), x)
# plot(x, y)
# savefig("preview.png")

dx = ComplexF32.([1e-4])
l_init = [5]
params = collect(Iterators.product(dx, l_init))

for p in params
	C = ConstraintFunction(2)
	o = optimize(C, z_init=ComplexF32(247.5949), dx = p[1], l_init=p[2], iter=5)
	println(o)
end
