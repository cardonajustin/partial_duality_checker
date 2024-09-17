include("constraint_function.jl")
using BaryRational


function find_zero(C::ConstraintFunction{T}; z_init::T=T(1.0), l_init::Int=10, domain=zeros(T, 1), codomain=zeros(real(T), 1), window::Int=10, points::Int=0, width::T=T(1)) where T
	err = typemax(real(T))
	z = z_init
	if length(domain) == 1
		# domain = width .* randn(real(T), l_init) .+ real(T)(z)
		domain = width .* (rand(real(T), l_init) .- 0.5) .+ real(T)(z)
		codomain = map(x-> C*T(x), domain)
	end
    
	i = 0
	while true
		err > cbrt(eps(real(T))) || break
		i += 1

		z = rand(real(T)) + real(T)(z - 0.5)
		window = min(window, length(domain))
		domain = vcat(domain, [z])[end-window:end]
		codomain = vcat(codomain, [C*T(z)])[end-window:end]
		
		a = aaa(domain, codomain, clean=1)
		_, _, zeros = prz(a)
		zeros = real.(zeros)
		try
			z1 = real(T)(zeros[end])
			z2 = real(T)(zeros[argmin(abs.(zeros .- z))])
			z = T(min(z1, z2))
			println("\t", i, ": ", norm(C*z), " at ", z)
			println("\t", zeros)
			readline()
			err = norm(C * z)
		catch
			println("\tRestarting after ", i, " iterations.")
			return find_zero(C, z_init=z, l_init=l_init, window=window, points=i)
		end
	end
	return real(z), err, domain, codomain, points+i
end


function optimize(C::ConstraintFunction{T}; l_init::Int=typemax(Int), z_init::T=T(1.0), iter::Int=10, dx::T=T(1e-3)) where T
	errs = Array{real(T)}(undef, iter, 2)
	for i in range(1, iter)
		println("Depth ", i)
		z, e, domain, codomain, p = find_zero(C, l_init=Int(floor(l_init * (0.5 + i/2))), z_init=z_init, width=dx)
		errs[i, :] = [e, real(T)(p)]
		perturb(C, dx)
	end
	return errs
end
