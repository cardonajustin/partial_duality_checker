using Serialization, Statistics, Plots


n_volume = 2
dx=1e-2
zs = deserialize("data/zeros_"*string(n_volume)*"_"*string(dx)*".dat")
ns = deserialize("data/iters_"*string(n_volume)*"_"*string(dx)*".dat")

ms = mean(ns, dims=1)'
ss = std(ns, dims=1)'
plot(LinRange(1, 3, 3), ms, ribbon=ss, fillalpha=.5, show=true)
readline()
histogram(ns[:], show=true)
readline()
@show mean(ns[:])
@show std(ns[:])