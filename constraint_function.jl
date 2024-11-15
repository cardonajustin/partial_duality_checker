include(ENV["MOLERING"]*"/gila/utils.jl")
using LinearAlgebra, CUDA, GilaElectromagnetics, .Utils, JacobiDavidson, Roots, BaryRational, ProgressMeter, Serialization


mutable struct ConstraintFunction{T<:Complex}
	G::AbstractMatrix{T}
	V::AbstractMatrix{T}
	O::AbstractMatrix{T}
	S::AbstractArray{T}
	R1::AbstractMatrix{T}
	R2::AbstractMatrix{T}
end


function ConstraintFunction(; n::Int=16, T::Type=ComplexF32, G::Union{Nothing, AbstractMatrix}=nothing, use_gpu=true)
	if G == nothing
		G = get_dense_G_matrix(load_greens_operator((n, n, n), (1//32, 1//32, 1//32), set_type=T, use_gpu=use_gpu))
	end
	m = size(G, 2)

	if use_gpu
		V = (rand(T) - T(0.5) + T(1e-3im)*rand(T)) .* CUDA.Diagonal(CUDA.ones(T, m))
		O = CUDA.Diagonal(T.(CUDA.rand(real(T), m)) .- T(0.5))
		R1 = CUDA.Diagonal(T.(CUDA.rand(real(T), m)))
		R2 = CUDA.Diagonal(T.(CUDA.rand(real(T), m)))
		S = CUDA.rand(T, m) .- T(0.5)
		return ConstraintFunction{T}(G, V, O, S, R1, R2)
	end
	#V = T(1e3) * Diagonal(ones(T, m) .- 0.5)
	#O = Diagonal(T.(ones(rand(T), m) .- 0.5))
	#R1 = Diagonal(T.(ones(rand(T), m)))
	#R2 = Diagonal(T.(ones(rand(T), m)))
	#S = ones(T, m)
	return ConstraintFunction{T}(G, V, O, S, R1, R2)
end


function get_ZTT_E(C::ConstraintFunction{T}) where T
	Sym = M::AbstractMatrix -> T(0.5) * (M + adjoint(M))
	ASym = M::AbstractMatrix -> T(-0.5im) * (M - adjoint(M))
	U = conj.(inv(C.V)) - adjoint(C.G)
	ZTT = C.O + Sym(U * C.R1) + ASym(U * C.R2)
	E = ASym(U)
	return ZTT, E
end


function zero_est(C::ConstraintFunction{T}) where T
	ZTT, E = get_ZTT_E(C)
	pschur, residuals = jdqz(Array(ZTT), Array(E), solver = GMRES(size(C.G, 2)), verbosity=1, pairs=1)
	found = pschur.alphas ./ pschur.betas
	if real(found[1]) > 0
		pschur, residuals = jdqz(Array(ZTT), Array(E), solver = GMRES(size(C.G, 2)), verbosity=1, target=Near(-found[1]))
		found = pschur.alphas ./ pschur.betas
	end
	z_min = -real(found[1])
	fx = ZeroProblem(x->C*T(x), (z_min, MathConstants.e^MathConstants.e * z_min))
	return T(solve(fx, Bisection(), verbose=false))
end


function eval(C::ConstraintFunction{T}, z::T) where T
	ZTT, E = get_ZTT_E(C)
	Z = ZTT + z * E
	ZTS = T(0.5) * (C.R1 + T(im) * (C.R2 + z * I))
	t = Z \ (ZTS * C.S)
	return real(imag(C.S' * t) - t' * E * t)
end
Base.:*(C::ConstraintFunction, z::Complex) = eval(C, z)
Base.eltype(C::ConstraintFunction) = eltype(C.G)
Base.size(C::ConstraintFunction) = size(C.G)


function find_zero(C::ConstraintFunction{T}; z_init::T=zero_est(C), domain::Union{Nothing, AbstractVector{T}}=nothing, codomain::Union{Nothing, AbstractVector{T}}=nothing, max_iter::Int=5, width::T=T(1e-2), n_init::Int=2, offset::Int=0) where T
		if offset > 100
			throw("did not converge")
		end
        if domain == nothing
                domain = width .* (rand(real(T), n_init) .- 0.5) .+ real(T)(z_init)
                codomain = map(x-> C*T(x), domain)
        end

        for i in 1:max_iter
                a = aaa(domain, codomain, clean=1)
                _, _, zeros = prz(a)
                r = 1 / (MathConstants.e)
                zeros = T.(collect(sort(filter(x-> x>real(z_init * (1-r)) && x<real(z_init * (1+r)), real.(zeros)))))
                length(zeros) > 0 || return find_zero(C, z_init=z_init, offset=n_init+i+offset)

                err = C * zeros[end]
                err > eps(real(T)) || return zeros[end], n_init + i + offset
                domain = vcat(domain, [zeros[end]])
                codomain = vcat(codomain, [err])
        end
        return find_zero(C, z_init=z_init, offset=n_init+max_iter+offset)
end
