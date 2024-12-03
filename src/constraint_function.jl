module constraint_function
include("utils.jl")
using .utils, GilaElectromagnetics, CUDA, LinearAlgebra, Roots, JacobiDavidson, BaryRational
export ConstraintFunction, get_LTT_E, partial_dual_root, pade_root

mutable struct ConstraintFunction
	G::GlaOpr
	X::AbstractMatrix{ComplexF64}
	Q::AbstractMatrix{ComplexF64}
	S::AbstractArray{ComplexF64}
	a::AbstractMatrix{ComplexF64}
	b::AbstractMatrix{ComplexF64}
end


function ConstraintFunction(n::Int=128)
    G = load_greens_operator((n, n, n), (1//32, 1//32, 1//32))
    m = size(G, 2)

    X = CUDA.Diagonal(ComplexF64(rand(Float64) - 0.5 + 1e-3im * rand(Float64)) .* CUDA.ones(ComplexF64, m))
    Q = CUDA.Diagonal(ComplexF64.(CUDA.rand(Float64, m)) .- ComplexF64(0.5))    
    S = CUDA.rand(ComplexF64, m) .- ComplexF64(0.5) #TODO offset imaginary too
    a = CUDA.Diagonal(ComplexF64.(CUDA.rand(Float64, m)))
    b = CUDA.Diagonal(ComplexF64.(CUDA.rand(Float64, m)))
    return ConstraintFunction(G, X, Q, S, a, b)
end


function get_LTT_E(C::ConstraintFunction, z::Float64)
    Sym = M::AbstractMatrix -> 0.5 * (M + adjoint(M))
    ASym = M::AbstractMatrix -> -0.5im * (M - adjoint(M))
    l = C.a - im*(C.b + z*I)
    LTT = x-> C.Q*x + Sym(adjoint(inv(C.X))*l)*x - 0.5 * (adjoint(C.G)*(l*x)) - 0.5 * (adjoint(l)*(C.G*x))
    E = x-> ASym(adjoint(inv(C.X)))*x + 0.5*im * (adjoint(C.G)*x - C.G*x)
    return LTT, E
end


function (C::ConstraintFunction)(z::Float64)
    LTT, E = get_LTT_E(C, z)
    LTS = 0.5 * (C.a + im * (C.b + z * I))
    T,_ = LTT \ (LTS * C.S)
    return real(imag(C.S' * T) - T' * E(T))
end


function partial_dual_root(C::ConstraintFunction)
    LTT, E = get_LTT_E(C, 0.0)
    #LTT = Array(fun_to_mat(LTT, size(C.G, 2)))
    #E = Array(fun_to_mat(E, size(C.G, 2)))
    #pschur, _ = jdqz(LTT, E, solver = GMRES(size(C.G, 2)), verbosity=0, pairs=1)
    #found = pschur.alphas ./ pschur.betas
    #if real(found[1]) > 0
    #    pschur, _ = jdqz(LTT, E, solver = GMRES(size(C.G, 2)), verbosity=0, pairs=1, target=Near(-found[1]))
    #    found = pschur.alphas ./ pschur.betas
    #end
    #if real(found[1]) > 0
    #    return 0.0
    #end
    #p = real(-found[1])
	g = powm_gpu(LTT, E, size(C.G, 2))[1]
	if real(g) > 0
		g = powm_gpu(LTT, E, size(C.G, 2), g)[1]
	end
	g = -real(g)
    fx = ZeroProblem(C, (g, 10 * g))
    #fx = ZeroProblem(C, (p, 10 * p))
    return solve(fx, Bisection(), verbose=false)
end


function pade_root(f, z_init::Float64; n_init::Int=1, max_iter::Int=5, max_restart::Int=2, r::Float64=1e-2, tol=eps(Float32))
    inverse_solves = 0
    err = 0
    for _ in 0:max_restart
        domain = r .* rand(Float64, n_init) .+ z_init .- 0.5
        domain = vcat(domain, [z_init])
        codomain = map(f, domain)
        inverse_solves += n_init
        for _ in 1:max_iter
            a = aaa(domain, codomain, clean=1)
            _, _, zeros = prz(a)
            z = maximum(real.(zeros))
            err = f(z)
            inverse_solves += 1
            abs(err) > tol || return z, inverse_solves
            println("\t\terr: "*string(err))
            domain = vcat(domain, [z])
            codomain = vcat(codomain, [err])
        end
        println("\tPPD did not converge, resampling")
		z_init = domain[argmin(abs.(codomain))]
    end
    throw("Pade Zero Finder did not converge")
end
end
