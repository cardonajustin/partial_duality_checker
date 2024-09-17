include(ENV["MOLERING"]*"/gila/utils.jl")
using LinearAlgebra, CUDA, GilaElectromagnetics, .Utils


mutable struct ConstraintFunction{T<:Complex}
	G::AbstractMatrix{T}
	V::AbstractMatrix{T}
	O::AbstractMatrix{T}
	S::AbstractArray{T}
	R1::AbstractMatrix{T}
	R2::AbstractMatrix{T}
end


function ConstraintFunction(n::Int; T=ComplexF32)
	G = get_dense_G_matrix(load_greens_operator((n, n, n), (1//32, 1//32, 1//32), set_type=T, use_gpu=true))
	size_G = size(G, 2)

	V = T(1e3) * CUDA.Diagonal(CUDA.rand(T, size_G) .- 0.5)
	O = CUDA.Diagonal(T.(CUDA.rand(real(T), size_G) .- 0.5))
	R1 = CUDA.Diagonal(T.(CUDA.rand(real(T), size_G)))
	R2 = CUDA.Diagonal(T.(CUDA.rand(real(T), size_G)))
	S = CUDA.rand(T, size_G)

	return ConstraintFunction{T}(G, V, O, S, R1, R2)
end


function eval(C::ConstraintFunction{T}, z::T) where T
	Sym = M::AbstractMatrix -> T(0.5) * (M + adjoint(M))
	ASym = M::AbstractMatrix -> T(-0.5im) * (M - adjoint(M))

	U = conj.(inv(C.V)) - adjoint(C.G)
	ZTT = C.O + Sym(U * C.R1) + ASym(U * C.R2)
	E = ASym(U)

	Z = ZTT + z * E
	ZTS = T(0.5) * (C.R1 + T(im) * C.R2)
	# t = bicgstab_gpu(Z, ZTS * C.S, max_iter=typemax(Int64), verbose=false)[1]
	t = Z \ (ZTS * C.S)
	return real(imag(C.S' * t) - t' * E * t)
end
Base.:*(C::ConstraintFunction, z::Complex) = eval(C, z)
Base.eltype(C::ConstraintFunction) = eltype(C.G)
Base.length(C::ConstraintFunction) = size(C.G, 1)


function perturb(C::ConstraintFunction{T}, dx::T) where T
	C.R1 += CUDA.Diagonal(T.(dx .* CUDA.rand(real(T), length(C))))
	C.R2 += CUDA.Diagonal(T.(dx .* CUDA.rand(real(T), length(C))))
end
