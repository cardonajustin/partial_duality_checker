include("../src/constraint_function.jl")
using .constraint_function
using CUDA, Plots, BaryRational


n = 2
r = 1e-1
C = ConstraintFunction(n)
root = partial_dual_root(C)

for _ in 1:100
	global root = pade_root(C, root)[1]
	@show root
	C.a += CUDA.Diagonal(r .* CUDA.randn(ComplexF64, size(C.a, 2)))
	C.b += CUDA.Diagonal(r .* CUDA.randn(ComplexF64, size(C.b, 2)))
end
