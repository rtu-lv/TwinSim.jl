using Statistics
using VisuTwinSim

positions = random_walk_ensemble(trajectories = 100_000, steps = 1_000, seed = 42)

println("trajectories = ", length(positions))
println("mean = ", mean(positions))
println("std = ", std(positions))
