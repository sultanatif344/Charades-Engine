using Statistics

function entropy_slope(trajectory::Vector{Float32})
    if length(trajectory) < 2
        return 0.0f0
    end

    n = length(trajectory)
    x = Float32.(1:n)

    x_mean = mean(x)
    y_mean = mean(trajectory)

    numerator = sum((x .- x_mean) .* (trajectory .- y_mean))
    denominator = sum((x .- x_mean) .^ 2)

    return denominator == 0 ? 0.0f0 : Float32(numerator / denominator)
end

function was_good_decision(slope::Float32)
    return slope < 0
end

struct DecisionLog
    context_fingerprint::String
    token_chosen::Int
    token_text::String
    avg_entropy_score::Float32
    coherence_score::Float32
    slope::Float32
    was_good::Bool
    timestamp::Float64
end

mutable struct Profiler
    decisions::Vector{DecisionLog}
    total_decisions::Int
    good_decisions::Int
end

Profiler() = Profiler(DecisionLog[], 0, 0)

function log_decision!(profiler::Profiler,
    context::String,
    token_id::Int,
    token_text::String,
    avg_entropy::Float32,
    coherence::Float32,
    post_trajectory::Vector{Float32})

    slope = entropy_slope(post_trajectory)
    good = was_good_decision(slope)

    entry = DecisionLog(
        context,
        token_id,
        token_text,
        avg_entropy,
        coherence,
        slope,
        good,
        time()
    )

    push!(profiler.decisions, entry)
    profiler.total_decisions += 1
    if good
        profiler.good_decisions += 1
    end

    return entry
end

function accuracy(profiler::Profiler)
    profiler.total_decisions == 0 && return 0.0f0
    return Float32(profiler.good_decisions / profiler.total_decisions)
end

function print_summary(profiler::Profiler)
    println("\n Profiler Summary")
    println("=" ^ 40)
    println("Total decisions: $(profiler.total_decisions)")
    println("Good decisions:  $(profiler.good_decisions)")
    println("Accuracy:        $(round(accuracy(profiler) * 100, digits=1))%")
    println()
    println("Decision log:")
    for d in profiler.decisions
        icon = d.was_good ? "✅" : "❌"
        println("$icon Context: '$(d.context_fingerprint)' → '$(d.token_text)'")
        println("   slope=$(round(d.slope, digits=4)) | entropy=$(round(d.avg_entropy_score, digits=2)) | coherence=$(round(d.coherence_score, digits=3))")
    end
end

function get_context_fingerprint(model, token_list::Vector{Int64}, position::Int)
    start = max(1, position - 3)
    context_tokens = token_list[start:position]

    words = String[]
    for id in context_tokens
        token_bytes = model.detokenize([id])
        token_text = pyconvert(String, token_bytes.decode("utf-8", errors="replace"))
        push!(words, strip(token_text))
    end

    return join(words, " ")
end