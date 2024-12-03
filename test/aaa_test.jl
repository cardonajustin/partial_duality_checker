include("../src/constraint_function.jl")
using .constraint_function
using CUDA, Plots, BaryRational


n = 2
r = 1e-1
C = ConstraintFunction(n)
@show "Made Constraint Function"
root = partial_dual_root(C)
domain = LinRange(root-r, root+r, 20)
codomain = map(C, domain)

samples = (root - 0.5) .+ (2*r .* rand(5))
cosamples = map(C, samples)
@show "made initial samples"
a = aaa(samples, cosamples)

plot(domain, map(C, domain))
plot!(domain, map(a, domain))
savefig("data/preview.png")
println("Press ENTER to start perturbing")
readline()

for _ in 1:3
	println("Press ENTER to perturb")
	readline()
	C.a += CUDA.Diagonal(1e-3 .* CUDA.randn(ComplexF64, size(C.a, 2)))
	C.b += CUDA.Diagonal(1e-3 .* CUDA.randn(ComplexF64, size(C.b, 2)))
	global domain = LinRange(root-r, root+r, 20)
	global codomain = map(C, domain)

	global samples = (root - 0.5) .+ (2*r .* rand(5))
	global cosamples = map(C, samples)
	global a = aaa(samples, cosamples)

	plot(domain, codomain)
	plot!(domain, map(a, domain))
	savefig("data/preview.png")
end
