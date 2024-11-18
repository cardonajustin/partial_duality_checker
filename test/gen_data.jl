include("../src/constraint_function.jl")
using .constraint_function, CUDA, ProgressMeter, Serialization


function zeros_perturbed(n_perturb::Int, n_volume::Int, dx::Float64=1e-2)
    zs = zeros(ComplexF64, n_perturb)
    ns = zeros(Int, n_perturb)
    C = ConstraintFunction(n_volume)
    g = partial_dual_root(C)
    for i in 1:n_perturb
		println("\t" * string(i) * " perturbations")
        C.P1 += dx .* CUDA.randn(ComplexF64, size(C.P1))
        C.P2 += dx .* CUDA.randn(ComplexF64, size(C.P2))
        result = pade_root(C, g)
        zs[i] = result[1]
        ns[i] = result[2]
        g = result[1]
    end
    return zs, ns    
end


function stat_test(n_samples::Int, n_perturb::Int, n_volume::Int, dx::Float64=1e-2)
    zs = zeros(ComplexF64, n_samples, n_perturb)
    ns = zeros(Int, n_samples, n_perturb)
    for i in 1:n_samples
		println("Sample " * string(i))
        while true
            try
                result = zeros_perturbed(n_perturb, n_volume, dx)
                zs[i, :] = result[1]
                ns[i, :] = result[2]
                break
            catch
				println("Restarting at sample " * string(i))
                continue
            end
        end
    end
    return zs, ns    
end


n_samples = 100
n_perturb = 10
n_volume = 2
dx=1e-3
zs, ns = stat_test(n_samples, n_perturb, n_volume, dx)
serialize("data/zeros_"*string(n_volume)*"_"*string(dx)*".dat", zs)
serialize("data/iters_"*string(n_volume)*"_"*string(dx)*".dat", ns)
