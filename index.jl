include("entropy.jl")
include("model.jl")
include("charades.jl")
include("profiler.jl")
include("visualizer.jl")

model = load_model("C:\\Users\\LENOVO\\Personal\\AI Models\\witty_v4.gguf")
embed_model = load_embedding_model("C:\\Users\\LENOVO\\Personal\\AI Models\\witty_v4.gguf")

hints = load_hints("hints.json")
cache = HintCache()
precompute_hint_embeddings!(cache, hints, embed_model)

profiler = Profiler()
println("Loaded $(length(hints)) hint scenes")

prompt = "def calculate_gradient(weights, learning_rate=0.01):"

t1 = @elapsed state, logits = generate_token_by_token(model, prompt, 1)
println("  Entropy scan:     $(round(t1, digits=2))s")

const MAX_TRIGGERS = 2

py_tokens = model.tokenize(pybuiltins.bytes(prompt, "utf-8"))
token_list = pyconvert(Vector{Int64}, py_tokens)

triggers = state.entropy_tracker.trigger_positions[1:min(MAX_TRIGGERS, length(state.entropy_tracker.trigger_positions))]

triggers = filter(p -> p > 2, state.entropy_tracker.trigger_positions)
triggers = triggers[1:min(MAX_TRIGGERS, length(triggers))]

for pos in triggers
    local uncertain_logits, active_tags, winner, trajectory, candidates, t2

    println("\n🎯 Processing uncertain position $pos")

    uncertain_logits = logits[pos, 1:end]
    partial_prompt = reconstruct_prompt_to_position(token_list, model, pos)
    active_tags = ask_witty_for_tags(model, prompt, hints)

    t2 = @elapsed winner, trajectory, candidates = charades(
        model, partial_prompt, uncertain_logits,
        active_tags, hints, embed_model, cache, :any
    )
    println("  Charades eval pos $pos: $(round(t2, digits=2))s")

    # Skip logging if charades found no good candidates
    if winner === nothing
        println("  ⏭️  Skipping profiler log - no quality candidates found")
        continue
    end

    context = get_context_fingerprint(model, token_list, pos)
    log_decision!(
        profiler, context,
        winner.token_id, winner.token_text,
        winner.avg_entropy, winner.coherence_score,
        trajectory
    )

    print_summary(profiler)
    save_all_plots(state.entropy_tracker, length(token_list), candidates, winner, profiler)
end

