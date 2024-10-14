using LinearAlgebra, CUDA, ProgressMeter, Statistics

function trace_h(A::AbstractMatrix{T}; tol_t::T=T(1e-3), tol_p::T=T(1e-3)) where T
	tol_t = real(T)(tol_t)
	tol_p = real(T)(tol_p)
	n = ceil(Int, (2 / tol_t^2) * (1 - (8/3)*tol_t)*log(1 / tol_p))
	
	t = 0
	p = Progress(n)
	for i in 1:n
		x = rand([T(-1), T(1)], size(A, 2))
		t += (x'*A*x - t) / i
		next!(p)
	end
	finish!(p)
	return t, n
end
