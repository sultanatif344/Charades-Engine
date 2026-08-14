using Statistics
using JSON3

const N_CANDIDATES = 3
const MAX_CANDIDATE_TOKENS = 25
const HINTS_PATH = "hints.json"


mutable struct HintCache
    anchor_embeddings::Dict{String,Vector{Float32}}
    elimination_embeddings::Dict{String,Vector{Float32}}
end

struct HintScene
    scene_id::String
    tags::Vector{String}
    coherence_anchors::Vector{String}
    elimination_criteria::Vector{String}
end

struct CandidateResult
    token_id::Int
    token_text::String
    avg_entropy::Float32
    entropy_trajectory::Vector{Float32}
    coherence_score::Float32
    post_decision_slope::Float32
end

function load_hints(path::String)
    raw = JSON3.read(read(path, String))
    scenes = HintScene[]
    for s in raw["scenes"]
        push!(scenes, HintScene(
            String(s["scene_id"]),
            String.(s["tags"]),
            String.(s["coherence_anchors"]),
            String.(s["elimination_criteria"])
        ))
    end
    return scenes
end


function match_any(scene::HintScene, active_tags::Vector{String})
    return any(t -> t in scene.tags, active_tags)
end

function match_all(scene::HintScene, active_tags::Vector{String})
    return all(t -> t in scene.tags, active_tags)
end

HintCache() = HintCache(Dict(), Dict())

function precompute_hint_embeddings!(cache::HintCache,
    hints::Vector{HintScene},
    embed_model)
    println("🔄 Precomputing hint embeddings...")

    total = sum(length(s.coherence_anchors) + length(s.elimination_criteria) for s in hints)
    count = 0

    for scene in hints
        for anchor in scene.coherence_anchors
            cache.anchor_embeddings[anchor] = get_embedding(embed_model, anchor)
            count += 1
            print("\r  Progress: $count/$total")
        end
        for criterion in scene.elimination_criteria
            cache.elimination_embeddings[criterion] = get_embedding(embed_model, criterion)
            count += 1
            print("\r  Progress: $count/$total")
        end
    end

    println("\n✅ Cached $count embeddings!")
    return cache
end

function get_relevant_hints(hints::Vector{HintScene}, active_tags::Vector{String}, strategy::Symbol)
    if strategy == :any
        return filter(s -> match_any(s, active_tags), hints)
    elseif strategy == :all
        return filter(s -> match_all(s, active_tags), hints)
    end
    return HintScene[]
end


function get_embedding(embed_model, text::String, cache::Union{Dict{String,Vector{Float32}},Nothing}=nothing)
    if cache !== nothing && haskey(cache, text)
        return cache[text]
    end

    result = embed_model.create_embedding(text)
    raw = result["data"][0]["embedding"]

    flat = Float32[]
    for chunk in raw
        for val in chunk
            push!(flat, Float32(pyconvert(Float64, val)))
        end
    end

    dims = 4096
    n_tokens = length(flat) ÷ dims

    pooled = zeros(Float32, dims)
    for i in 1:n_tokens
        start_idx = (i-1) * dims + 1
        end_idx = i * dims
        pooled .+= flat[start_idx:end_idx]
    end
    pooled ./= n_tokens


    return pooled
end


function cosine_similarity(a::Vector{Float32}, b::Vector{Float32})
    dot_product = sum(a .* b)
    norm_a = sqrt(sum(a .^ 2))
    norm_b = sqrt(sum(b .^ 2))
    if norm_a == 0 || norm_b == 0
        return 0.0f0
    end
    return Float32(dot_product / (norm_a * norm_b))
end

function coherence_score_embedding(continuation_text::String,
    relevant_hints::Vector{HintScene},
    embed_model, cache::HintCache)
    if isempty(relevant_hints)
        return 0.5f0
    end

    continuation_embedding = get_embedding(embed_model, continuation_text)

    anchor_score = 0.0f0
    elimination_score = 0.0f0
    total_anchors = 0
    total_eliminations = 0

    for scene in relevant_hints
        for anchor in scene.coherence_anchors
            total_anchors += 1
            anchor_embedding = get_embedding(embed_model, anchor, cache.anchor_embeddings)
            sim = cosine_similarity(continuation_embedding, anchor_embedding)
            anchor_score += sim
        end
        for criterion in scene.elimination_criteria
            total_eliminations += 1
            criterion_embedding = get_embedding(embed_model, criterion, cache.elimination_embeddings)
            sim = cosine_similarity(continuation_embedding, criterion_embedding)
            elimination_score += sim
        end
    end

    anchor_ratio = total_anchors > 0 ? Float32(anchor_score / total_anchors) : 0.0f0
    elimination_ratio = total_eliminations > 0 ? Float32(elimination_score / total_eliminations) : 0.0f0

    return anchor_ratio - elimination_ratio
end

function evaluate_candidate(model, prompt::String, candidate_token_id::Int,
    relevant_hints::Vector{HintScene}, embed_model, cache::HintCache)

    token_bytes = model.detokenize([candidate_token_id])
    token_text = pyconvert(String, token_bytes.decode("utf-8", errors="replace"))
    candidate_prompt = prompt * token_text

    # Generate continuation
    output = model(candidate_prompt, max_tokens=MAX_CANDIDATE_TOKENS, echo=false)
    generated = pyconvert(String, output["choices"][0]["text"])
    println("    Continuation: '$generated'")

    # Entropy trajectory
    model(candidate_prompt, max_tokens=MAX_CANDIDATE_TOKENS, echo=true)
    raw_logits = model.scores
    logits_matrix = copy(pyconvert(Array{Float32}, raw_logits))

    py_tokens = model.tokenize(pybuiltins.bytes(candidate_prompt, "utf-8"))
    token_list = pyconvert(Vector{Int64}, py_tokens)
    actual_length = length(token_list)

    original_length = max(1, actual_length - MAX_CANDIDATE_TOKENS)
    trajectory = Float32[]

    for pos in original_length:actual_length
        current_logits = logits_matrix[pos, 1:end]
        e = entropy(current_logits)
        push!(trajectory, e)
    end

    avg = isempty(trajectory) ? Float32(10.4) : Float32(mean(trajectory))
    slope = entropy_slope(trajectory)
    cscore = coherence_score_embedding(generated, relevant_hints, embed_model, cache)

    return CandidateResult(
        candidate_token_id,
        token_text,
        avg,
        trajectory,
        cscore,
        slope
    )
end

function detect_tags_from_embeddings(prompt::String, hints::Vector{HintScene}, embed_model, cache::HintCache, top_k::Int=2)
    prompt_embedding = get_embedding(embed_model, prompt)
    # Don't cache this one - prompts are always unique

    scene_scores = Float32[]
    for scene in hints
        scores = Float32[]
        for anchor in scene.coherence_anchors
            anchor_embedding = get_embedding(embed_model, anchor, cache.anchor_embeddings)
            sim = cosine_similarity(prompt_embedding, anchor_embedding)
            push!(scores, sim)
        end
        push!(scene_scores, Float32(mean(scores)))
    end

    top_indices = sortperm(scene_scores, rev=true)[1:min(top_k, length(hints))]

    detected_tags = String[]
    for idx in top_indices
        append!(detected_tags, hints[idx].tags)
    end

    return unique(detected_tags)
end


function ask_witty_for_tags(model, prompt::String, hints::Vector{HintScene})
    # Build list of all known tags from hints.json
    all_tags = unique(vcat([scene.tags for scene in hints]...))
    tags_list = join(all_tags, ", ")

    # Ask Witty to identify the context
    tag_prompt = """Identify the genre and tone tags for this scene.

Scene: $prompt

Available tags: $tags_list

Tags that apply:"""

    output = model(tag_prompt, max_tokens=20, echo=false)
    raw = pyconvert(String, output["choices"][0]["text"])

    println("🏷️  Witty's raw tag response: '$raw'")

    # Parse the response - look for known tags in the output
    detected = String[]
    raw_lower = lowercase(raw)

    for tag in all_tags
        if occursin(lowercase(tag), raw_lower)
            push!(detected, tag)
        end
    end

    # Fall back to nearest neighbor if Witty doesn't recognize context
    if isempty(detected)
        println("  ⚠️  No tags detected from Witty, falling back to nearest neighbor")
        detected = detect_tags_from_embeddings(prompt, hints, embed_model, cache)
    end

    println("  Active tags: $detected")
    return detected
end

function softmax_combined_score(results::Vector{CandidateResult})
    entropies = Float32[r.avg_entropy for r in results]
    coherences = Float32[r.coherence_score for r in results]
    slopes = Float32[r.post_decision_slope for r in results]

    # Reuse existing softmax from entropy.jl
    # Negate entropy and slope since lower is better
    prob_entropy = softmax(-entropies)
    prob_coherence = softmax(coherences)
    prob_slope = softmax(-slopes)

    # Combined - higher total probability = better candidate
    return prob_entropy .+ prob_coherence .+ prob_slope
end

function candidates_worth_evaluating(top_ids::Vector{Int},
    model,
    relevant_hints::Vector{HintScene},
    cache::HintCache,
    embed_model)

    if isempty(relevant_hints)
        return true  # no hints to check against, proceed anyway
    end

    meaningful_tokens = 0

    for id in top_ids
        token_bytes = model.detokenize([id])
        token_text = pyconvert(String, token_bytes.decode("utf-8", errors="replace"))
        cleaned = strip(token_text)

        # Skip empty, single char, or punctuation tokens
        # These don't carry enough semantic meaning for reliable pre-filtering
        if length(cleaned) <= 1 || all(c -> ispunct(c), cleaned)
            println("⚡ Skipping trivial token: '$(cleaned)'")
            continue
        end

        meaningful_tokens += 1

        token_embedding = get_embedding(embed_model, token_text)

        # Check against all anchor phrases in relevant scenes
        for scene in relevant_hints
            for anchor in scene.coherence_anchors
                anchor_emb = get_embedding(embed_model, anchor, cache.anchor_embeddings)
                sim = cosine_similarity(token_embedding, anchor_emb)
                if sim > 0.2  # meaningful positive signal
                    println("⚡ Pre-filter passed: '$(strip(token_text))' similarity $(round(sim, digits=3)) with '$(anchor[1:min(30,length(anchor))])'")
                    return true
                end
            end
        end
    end

    # If all tokens were trivial fall through to full evaluation
    if meaningful_tokens == 0
        println("⚡ All tokens trivial - proceeding to full evaluation")
        return true
    end

    println("⚡ Pre-filter: all meaningful candidates outside known territory")
    return false
end

function charades(model, prompt::String, logits_at_position::Vector{Float32},
    active_tags::Vector{String}, hints::Vector{HintScene},
    embed_model, cache::HintCache, strategy::Symbol=:any)

    println("Available threads: $(Threads.nthreads())")
    println("\n🎭 Charades triggered!")
    println("Active tags: $active_tags")
    println("Matching strategy: $strategy")

    relevant_hints = get_relevant_hints(hints, active_tags, strategy)
    println("Matched $(length(relevant_hints)) hint scenes")

    top_ids = sortperm(logits_at_position, rev=true)[1:N_CANDIDATES]

    if !candidates_worth_evaluating(top_ids, model, relevant_hints, cache, embed_model)
        return nothing, Float32[], CandidateResult[]
    end

    results = CandidateResult[]

    for id in top_ids
        result = evaluate_candidate(model, prompt, id, relevant_hints, embed_model, cache)
        println("  '$(result.token_text)' → avg_entropy: $(round(result.avg_entropy, digits=2)) | coherence: $(round(result.coherence_score, digits=3)) | slope: $(round(result.post_decision_slope, digits=4))")
        push!(results, result)
    end

    all_negative = all(r -> r.coherence_score < 0.0, results)
    all_positive_slope = all(r -> r.post_decision_slope > 0.3, results)

    if all_negative || all_positive_slope
        println("\n⚠️  All candidates below quality threshold - skipping correction")
        println("   Reason: $(all_negative ? "all coherence negative" : "all slopes strongly positive")")
        return nothing, Float32[], results
    end

    scores = softmax_combined_score(results)
    best = results[argmax(scores)]

    winner_coherence_too_low = best.coherence_score < 0.0 && best.post_decision_slope < -0.2

    if winner_coherence_too_low
        println("⚠️  Winner confidently wrong - trying next best candidate")

        # Filter out the bad winner
        valid_results = filter(r -> !(r.coherence_score < 0.0 &&
                                      r.post_decision_slope < -0.2), results)

        if isempty(valid_results)
            println("⚠️  No valid candidates - graceful degradation")
            return nothing, Float32[], results
        end

        # Pick best from remaining
        valid_scores = softmax_combined_score(valid_results)
        best = valid_results[argmax(valid_scores)]
        println("  Fallback winner: '$(best.token_text)'")
    end

    for (r, s) in zip(results, scores)
        println("    '$(r.token_text)' → $(round(s, digits=4))")
    end
    println("\n✅ Winner: '$(best.token_text)'")
    println("   avg_entropy: $(round(best.avg_entropy, digits=2))")
    println("   coherence_score: $(round(best.coherence_score, digits=3))")
    println("   post_decision_slope: $(round(best.post_decision_slope, digits=4))")

    return best, best.entropy_trajectory, results
end