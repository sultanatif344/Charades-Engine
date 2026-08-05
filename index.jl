include("entropy.jl")
include("model.jl")
include("charades.jl")
include("profiler.jl")
include("visualizer.jl")

model = load_model("C:\\Users\\LENOVO\\Personal\\AI Models\\witty.gguf")
embed_model = load_embedding_model("C:\\Users\\LENOVO\\Personal\\AI Models\\witty.gguf")

hints = load_hints("hints.json")
profiler = Profiler()
println("Loaded $(length(hints)) hint scenes")

active_tags = ["dark_comedy", "unlikely_bond", "heartbreak"]

t1 = @elapsed state, logits = generate_token_by_token(
    model,
    "The door opened slowly. Neither of them spoke as",
    1
)

println("  Entropy scan:     $(round(t1, digits=2))s")

uncertain_logits = logits[2, 1:end]
t2 = @elapsed winner, trajectory, candidates = charades(model, "The door opened slowly. Neither of them spoke as",
    uncertain_logits, active_tags, hints, embed_model, :any)

println("  Charades eval:    $(round(t2, digits=2))s")


py_tokens = model.tokenize(pybuiltins.bytes("The door opened slowly. Neither of them spoke as", "utf-8"))
token_list = pyconvert(Vector{Int64}, py_tokens)
context = get_context_fingerprint(model, token_list, 2)

t3 = @elapsed entry = log_decision!(
    profiler,
    context,
    winner.token_id,
    winner.token_text,
    winner.avg_entropy,
    winner.coherence_score,
    trajectory
)

println("  Profiler log:     $(round(t3, digits=4))s")
println("  Total:            $(round(t1+t2+t3, digits=2))s")

print_summary(profiler)


save_all_plots(state.entropy_tracker, 11, candidates, winner, profiler)


