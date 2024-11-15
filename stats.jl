using Serialization, Plots, Statistics
ENV["GKSwstype"] = "100"

n = ARGS[1]
d = deserialize("data_$(n).dat")
data = Array(d)
points = Int.(data[:, 2])
println(mean(points))
println(std(points))
histogram(points, legend=false)


savefig("points_$(n).png")
