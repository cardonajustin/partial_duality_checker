include("constraint_function.jl")
using BaryRational, ProgressMeter


function find_zero(C::ConstraintFunction{T}; z_init::T=T(1.0), l_init::Int=typemax(Int), inc::Int=1, domain=zeros(T, 1), codomain=zeros(real(T), 1)) where T
	err = typemax(real(T))
	z = real(z_init)
	if length(domain) == 1
		domain = rand(real(T), floor(Int, length(C))) .+ z
		codomain = map(x-> C*T(x), domain)
	end

	@showprogress desc=norm(C*z) for _ in range(1, l_init)
		domain_add = rand(real(T), floor(Int, inc)) .+ z
		domain = vcat(domain, domain_add)
		codomain = vcat(codomain, map(x-> C*T(x), domain_add))
		
		a = aaa(domain, codomain, clean=1)
		_, _, zeros = prz(a)
		z = T(maximum(real.(zeros)))
		if norm(C * z) < sqrt(eps(real(T)))
			break
		else
			println(norm(C*z))
		end
	end
	return real(z), err, domain, codomain
end


function optimize(C::ConstraintFunction{T}; z_init::T=T(1.0), inc::Int=1, iter::Int=10) where T
	errs = Array{real(T)}(undef, iter, 2)
	z, e, domain, codomain = find_zero(C, z_init=z_init, inc=inc)
	for i in range(1, iter)
		println("Depth ", i)
		errs[i, :] = [e, length(domain)]
		z, e, domain, codomain = find_zero(C, z_init=T(z), l_init=1, inc=inc, domain=domain, codomain=codomain)
		perturb(C, T(1e-3))
	end
	return errs
end

