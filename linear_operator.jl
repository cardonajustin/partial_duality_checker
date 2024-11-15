include(ENV["MOLERING"]*"/gila/utils.jl")
using LinearAlgebra, GilaElectromagnetics, .Utils, CUDA
mutable struct LinearOperator
	func::Function
end

function LinearOperator(A)
	return LinearOperator(x-> A*x)
end

const T = Union{LinearOperator, GlaOpr, AbstractMatrix}
Base.:*(A::LinearOperator, x::AbstractVector) = A.func(x)
Base.:*(A::T, B::T) = LinearOperator(x -> A * (B * x))
Base.:+(A::T, B::T) = LinearOperator(x -> (A * x) + (B * x))
Base.:-(A::T, B::T) = LinearOperator(x -> (A * x) - (B * x))
