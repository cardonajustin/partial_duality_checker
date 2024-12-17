using Serialization, Statistics, Plots


n_volume = 2
dx=1e-3
zs = deserialize("data/zeros_"*string(n_volume)*"_"*string(dx)*".dat")
ns = deserialize("data/iters_"*string(n_volume)*"_"*string(dx)*".dat")

ms = mean(ns, dims=1)'
ss = std(ns, dims=1)'
plot(LinRange(1, 10, 10), ms, ribbon=ss, fillalpha=.5, legend=false, xlabel="Number of Perturbations", ylabel="Expected Number of Inverse Solves")
savefig("data/perturbation_iters.png")
histogram(ns[:], bins=5:1:25, yaxis = (:log10), legend=false)
savefig("data/histogram.png")
@show mean(ns[:])
@show std(ns[:])
