include("constraint_function.jl")


n = parse(Int, ARGS[1])
use_gpu = false
const T = ComplexF64
samples = 100
data = Array{T}(undef, samples, 2)
G = get_dense_G_matrix(load_greens_operator((n, n, n), (1//32, 1//32, 1//32), set_type=T, use_gpu=use_gpu))
println("Constructed Green's Operator")


pb = Progress(samples)
i = 1
global redo = false
while i <= samples
	try
		C = ConstraintFunction(G=G, use_gpu=use_gpu)
		data[i, :] .= T.(find_zero(C, max_iter=20))
		global redo = false
	catch
		global redo = true
	end
	if redo == false
		global i += 1
		next!(pb)
	end
end
finish!(pb)
serialize("data_$(n).dat", data)
