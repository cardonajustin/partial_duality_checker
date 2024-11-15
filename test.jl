include("constraint_function.jl")
using Plots, Statistics, ProgressMeter
ENV["GKSwstype"] = "100"

const T = ComplexF64
results = zeros(Int, 1000)
for i in 1:length(results)
	@show i
	while true
		try
			C = ConstraintFunction(n=4, T=T)
			global results[i] = find_zero(C)[2]
		catch
			continue
		end
		break
	end
end

@show mean(results)
@show std(results)
histogram(results, legend=false)
savefig("histogram.png")
