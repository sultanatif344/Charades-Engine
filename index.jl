include("entropy.jl")
include("model.jl")
include("charades.jl")
include("profiler.jl")
include("api.jl")
# include("visualizer.jl")

model_path = get(ENV, "WITTY_MODEL_PATH",
    "C:\\Users\\LENOVO\\Personal\\AI Models\\witty_v4.gguf")

println("Loading model from: $model_path")
model = load_model(model_path)

hints = load_hints("hints.json")
cache = HintCache()
precompute_hint_embeddings!(cache, hints, model)

profiler = Profiler()
println("Loaded $(length(hints)) hint scenes")

println("✅ Witty v4 ready!")

# Start API server
start_api()



