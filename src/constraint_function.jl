module constraint_function
include("utils.jl")
using .utils, GilaElectromagnetics, CUDA, LinearAlgebra, Roots, JacobiDavidson, BaryRational
export ConstraintFunction, get_ZTT_E, partial_dual_root, pade_root

mutable struct ConstraintFunction
	G::GlaOpr
	V::AbstractMatrix{ComplexF64}
	O::AbstractMatrix{ComplexF64}
	S::AbstractArray{ComplexF64}
	P1::AbstractMatrix{ComplexF64}
	P2::AbstractMatrix{ComplexF64}
end


function ConstraintFunction(n::Int=128)
    G = load_greens_operator((n, n, n), (1//32, 1//32, 1//32))
    m = size(G, 2)

    V = ComplexF64(rand(Float64) - 0.5 + 1e-3im * rand(Float64)) .* CUDA.Diagonal(CUDA.ones(ComplexF64, m))
    O = CUDA.Diagonal(ComplexF64.(CUDA.rand(Float64, m)) .- ComplexF64(0.5))    
    S = CUDA.rand(ComplexF64, m) .- ComplexF64(0.5) #TODO offset imaginary too
    P1 = CUDA.Diagonal(ComplexF64.(CUDA.rand(Float64, m)))
    P2 = CUDA.Diagonal(ComplexF64.(CUDA.rand(Float64, m)))
    return ConstraintFunction(G, V, O, S, P1, P2)
end


function get_ZTT_E(C::ConstraintFunction, z::Float64)
    Sym = M::AbstractMatrix -> 0.5 * (M + adjoint(M))
    ASym = M::AbstractMatrix -> -0.5im * (M - adjoint(M))
    P = C.P1 - im*(C.P2 + z*I)
    ZTT = x-> C.O*x + Sym(adjoint(inv(C.V))*P)*x - 0.5 * (adjoint(C.G)*(P*x)) - 0.5 * (adjoint(P)*(C.G*x))
    E = x-> ASym(adjoint(inv(C.V)))*x + 0.5*im * (adjoint(C.G)*x - C.G*x)
    return ZTT, E
end


function (C::ConstraintFunction)(z::Float64)
    ZTT, E = get_ZTT_E(C, z)
    ZTS = 0.5 * (C.P1 + im * (C.P2 + z * I))
    T = ZTT \ ZTS * C.S
    return real(imag(C.S' * T) - T' * E(T))
end


function partial_dual_root(C::ConstraintFunction)
    ZTT, E = get_ZTT_E(C, 0.0)
    ZTT = Array(fun_to_mat(ZTT, size(C.G, 2)))
    E = Array(fun_to_mat(E, size(C.G, 2)))
    pschur, _ = jdqz(ZTT, E, solver = GMRES(size(C.G, 2)), verbosity=0, pairs=1)
    found = pschur.alphas ./ pschur.betas
    if real(found[1]) > 0
        pschur, _ = jdqz(ZTT, E, solver = GMRES(size(C.G, 2)), verbosity=0, pairs=1, target=Near(-found[1]))
        found = pschur.alphas ./ pschur.betas
    end
    if real(found[1]) > 0
        return 0.0
    end
    p = real(-found[1])
    fx = ZeroProblem(C, (p, 10 * p))
    return solve(fx, Bisection(), verbose=false)
end


function pade_root(f, z_init::Float64; n_init::Int=2, max_iter::Int=5, r::Float64=1e-2, tol=eps(Float32))
    domain = r .* rand(Float64, n_init) .+ z_init .- 0.5
    codomain = map(f, domain)
    inverse_solves = n_init
    while true
        for i in 1:max_iter
            a = aaa(domain, codomain, clean=1)
            _, _, zeros = prz(a)
            z = maximum(real.(zeros))
            err = f(z)
            inverse_solves += 1
            abs(err) > tol || return z, inverse_solves
            domain = vcat(domain, [z])
            codomain = vcat(codomain, [err])
        end
    end
end
end