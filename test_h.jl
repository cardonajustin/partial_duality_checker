include("dual_check.jl")

n = 16
C = ConstraintFunction(n)
A = normalize!(Array(get_ZTT(C)))

#m = 3 * n^3
#A = rand(m, m)
#A = 0.5 * (A + A')
#U = svd(A).U
#D = Diagonal(10.0 .^ rand(-2:0, m))
#A = U' * D * U


function log_mat(A::AbstractMatrix{T}; n::Int=25) where T
	B = I - A
	result = zeros(T, size(A))
	for k in 1:n
		result += (1/k) * B
		B *= I - A
	end
	return -result
end


function trace_log(A::AbstractMatrix{T}; n::Int=25) where T
	result = 0
	B = I
	for k in 1:n
		B = B * (I - A) # can be faster by doing mvps in trace_h
		t = trace_h(B)
		result += (1/k) * t[1]
	end
	return -result
end


@show find_zero(C)[1]
#@show sum(diag(log(A)))
#@show sum(diag(log_mat(A)))
#@show trace_log(A)
