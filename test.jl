include("dual_check.jl")
using Plots

C = ConstraintFunction(2)
x = ComplexF32.(range(1, 10))
y = map(z-> C * z, x)
plot(Float32.(x), y)
savefig("preview.png")
# optimize(C)
