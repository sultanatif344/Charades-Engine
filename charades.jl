using Statistics
using JSON3

const N_CANDIDATES = 3
const MAX_CANDIDATE_TOKENS = 10
const HINTS_PATH = "hints.json"

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

function get_relevant_hints(hints::Vector{HintScene}, active_tags::Vector{String}, strategy::Symbol)
    if strategy == :any
        return filter(s -> match_any(s, active_tags), hints)
    elseif strategy == :all
        return filter(s -> match_all(s, active_tags), hints)
    end
    return HintScene[]
end

function load_embedding_model(model_path::String)
    llama_cpp = pyimport("llama_cpp")
    return llama_cpp.Llama(
        model_path=model_path,
        embedding=true,
        n_ctx=512,
        verbose=false
    )
end

function get_embedding(embed_model, text::String)
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
    embed_model)
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
            anchor_embedding = get_embedding(embed_model, anchor)
            sim = cosine_similarity(continuation_embedding, anchor_embedding)
            anchor_score += sim
        end
        for criterion in scene.elimination_criteria
            total_eliminations += 1
            criterion_embedding = get_embedding(embed_model, criterion)
            sim = cosine_similarity(continuation_embedding, criterion_embedding)
            elimination_score += sim
        end
    end

    anchor_ratio = total_anchors > 0 ? Float32(anchor_score / total_anchors) : 0.0f0
    elimination_ratio = total_eliminations > 0 ? Float32(elimination_score / total_eliminations) : 0.0f0

    return anchor_ratio - elimination_ratio
end

function evaluate_candidate(model, prompt::String, candidate_token_id::Int,
    relevant_hints::Vector{HintScene}, embed_model)

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

    cscore = coherence_score_embedding(generated, relevant_hints, embed_model)

    return CandidateResult(
        candidate_token_id,
        token_text,
        avg,
        trajectory,
        cscore
    )
end



function charades(model, prompt::String, logits_at_position::Vector{Float32},
    active_tags::Vector{String}, hints::Vector{HintScene},
    embed_model, strategy::Symbol=:any)

    println("\n🎭 Charades triggered!")
    println("Active tags: $active_tags")
    println("Matching strategy: $strategy")

    relevant_hints = get_relevant_hints(hints, active_tags, strategy)
    println("Matched $(length(relevant_hints)) hint scenes")

    top_ids = sortperm(logits_at_position, rev=true)[1:N_CANDIDATES]
    results = CandidateResult[]

    for id in top_ids
        result = evaluate_candidate(model, prompt, id, relevant_hints, embed_model)
        println("  '$(result.token_text)' → avg_entropy: $(round(result.avg_entropy, digits=2)) | coherence: $(round(result.coherence_score, digits=3))")
        push!(results, result)
    end

    best = results[argmin([r.avg_entropy - r.coherence_score for r in results])]

    println("\n Winner: '$(best.token_text)'")
    println("   avg_entropy: $(round(best.avg_entropy, digits=2))")
    println("   coherence_score: $(round(best.coherence_score, digits=3))")

    return best, best.entropy_trajectory, results
end