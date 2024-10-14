include("constraint_function.jl")
using BaryRational


function find_zero(C::ConstraintFunction{T}; z_init::T=zero_est(C), l_init::Int=10, domain=zeros(T, 1), codomain=zeros(real(T), 1), window::Int=20, points::Int=0, width::T=T(0.01), max_iter::Int=30) where T
	err = typemax(real(T))
	z = z_init
	if length(domain) == 1
		domain = width .* (rand(real(T), l_init) .- 0.5) .+ real(T)(z)
		codomain = map(x-> C*T(x), domain)
		ages = zeros(Int, l_init)
	else
		ages = zeros(Int, size(domain))
	end
    
	i = 0
	while i<=max_iter
		err > cbrt(eps(real(T))) || break
		i += 1

		z = rand(real(T)) + real(T)(z - 0.5)
		window = min(window, length(domain))
		domain = vcat(domain, [z])[end-window:end]
		codomain = vcat(codomain, [C*T(z)])[end-window:end]
		ages = vcat(ages .+ 1, [0])

		a = aaa(domain, codomain, clean=1)
		_, _, zeros = prz(a)
		r = 1 / (MathConstants.e)
		zeros = filter(x-> x>real(z_init * (1-r)) && x<real(z_init * (1+r)), real.(zeros))
		try
			z = T(zeros[1])
			err = norm(C * z)
			# println("\ti: ", i, " ", err)
		catch
			# println("\tRestarting after ", i, " iterations.")
			return find_zero(C, z_init=T(z), l_init=l_init, window=window, points=i)
		end
		i < max_iter || throw("did not converge")
	end
	return real(z), err, domain, codomain, points+i
end


function optimize(C::ConstraintFunction{T}; l_init::Int=typemax(Int), z_init::T=T(1.0), iter::Int=10, dx::T=T(1e-3), window::Int=20) where T
	errs = Array{real(T)}(undef, iter, 2)
	z, e, domain, codomain, p = find_zero(C, l_init=l_init, z_init=z_init, width=dx)
	errs[1, :] = [e, real(T)(p)]
	for i in range(2, iter)
		perturb(C, dx)
		z, e, domain, codomain, p = find_zero(C, z_init=z_init, domain=domain, codomain=codomain, window=window, width=dx)
		errs[i, :] = [e, real(T)(p)]
	end
	return z, errs
end
