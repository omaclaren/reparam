using Pkg
Pkg.activate(".")

packages = [
    "Distributions",
    "ForwardDiff",
    "NLopt",
    "Plots",
    "LaTeXStrings",
    "Measures",
    "DifferentialEquations",
    "Distributed",
    "Serialization",
    "CSV",
    "DataFrames"
]

println("Adding packages...")
for pkg in packages
    try
        Pkg.add(pkg)
        println("Added $pkg")
    catch e
        println("Failed to add $pkg: $e")
    end
end

println("Instantiating to generate Manifest.toml...")
Pkg.instantiate()
println("Setup complete. Project.toml and Manifest.toml are ready.")
