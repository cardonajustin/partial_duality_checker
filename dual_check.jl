include("constraint_function.jl")
using BaryRational


function find_zero(C::ConstraintFunction{T}, z_init::T; increment::Int=1) where T
	err = typemax(real(T))
	z = real(z_init)
	domain = rand(real(T), floor(Int, log(length(C)))) .+ z
	codomain = map(x-> C*T(x), domain)

	while norm(err) > sqrt(eps(real(T)))
		domain_add = rand(real(T), increment) .+ z
		domain = vcat(domain, domain_add)
		codomain = vcat(codomain, map(x-> C*T(x), domain_add))
		
		a = aaa(domain, codomain, clean=1)
		_, _, zeros = prz(a)
		z = T(maximum(real.(zeros)))
		err = C * z
	end
	return real(z), err
end
C = ConstraintFunction(2)
print(find_zero(C, eltype(C)(1.0)))
