include("entropy.jl")
include("model.jl")
include("charades.jl")
include("profiler.jl")
include("api.jl")
# include("visualizer.jl")

model = load_model("C:\\Users\\LENOVO\\Personal\\AI Models\\witty_v4.gguf")

hints = load_hints("hints.json")
cache = HintCache()
precompute_hint_embeddings!(cache, hints, model)

profiler = Profiler()
println("Loaded $(length(hints)) hint scenes")

println("✅ Witty v4 ready!")

# Start API server
start_api()



