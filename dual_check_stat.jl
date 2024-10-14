include("constraint_function.jl")
using BaryRational


function find_zero(C::ConstraintFunction{T}; z_init::T=T(1.0), l_init::Int=10, domain=zeros(T, 1), codomain=zeros(real(T), 1), ages=zeros(Int, 1), window::Int=20, points::Int=0, width::T=T(1)) where T
	err = typemax(real(T))
	z = z_init
	if length(domain) == 1
		domain = width .* (rand(real(T), l_init) .- 0.5) .+ real(T)(z)
		codomain = map(x-> C*T(x), domain)
		ages = zeros(Int, l_init)
	end
    
	i = 0
	while true
		err > sqrt(eps(real(T))) || break
		i += 1
		println("\ti: ", i)

		z = rand(real(T)) + real(T)(z - 0.5)
		window = min(window, length(domain))
		domain = vcat(domain, [z])[end-window:end]
		codomain = vcat(codomain, [C*T(z)])[end-window:end]
		ages = vcat(ages .+ 1, [0])[end-window:end]

		# reset unstable(left of zero guess) points
		idx = findall(x -> x<real(z_init), real.(domain))
		codomain[idx] = map(x-> C*T(x), domain[idx])
		ages[idx] = map(x->0, ages[idx])

		z_err = typemax(real(T))
		zs = []
		while true
			(z_err < cbrt(eps(real(T))) || length(zs) < 2) || break
			codomain_test = (codomain .- 0.5) + 0.05 .* ages .* codomain .* rand(T, size(codomain))
			a = aaa(domain, codomain_test, clean=1)
			_, _, zeros = prz(a)
			r = 1 / (2 * MathConstants.e)
			r = 1 / (MathConstants.e)
			zeros = filter(x-> x>real(z_init * (1-r)) && x<real(z_init * (1+r)), real.(zeros))
			zs = vcat(zs, [T(zeros[1])])
			z_err = var(zs)
			println("\t", mean(zs), " ", z_err / length(zs))
		end
		z = T(mean(z))
		err = norm(C * z)
	end
	return real(z), err, domain, codomain, ages, points+i
end


function optimize(C::ConstraintFunction{T}; l_init::Int=typemax(Int), z_init::T=T(1.0), iter::Int=10, dx::T=T(1e-3), window::Int=20) where T
	errs = Array{real(T)}(undef, iter, 2)
	z, e, domain, codomain, ages, p = find_zero(C, l_init=l_init, z_init=z_init, width=dx)
	errs[1, :] = [e, real(T)(p)]
	for i in range(2, iter)
		perturb(C, dx)
		println("Depth ", i)
		z, e, domain, codomain, ages, p = find_zero(C, z_init=z_init, domain=domain, codomain=codomain, ages=ages, window=window, width=dx)
		errs[i, :] = [e, real(T)(p)]
	end
	return z, errs
end
