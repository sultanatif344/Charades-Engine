using PythonCall

function softmax(x::Vector{Float32})
    e = exp.(x .- maximum(x))
    return e ./ sum(e)
end

function entropy(logits_row::Vector{Float32})
    probs = softmax(logits_row)
    return Float32(-sum(p * log(p + 1e-10) for p in probs))
end

const ENTROPY_THRESHOLD = 5.0

function is_uncertain(e::Real)
    return e > ENTROPY_THRESHOLD
end

function top_n_tokens(logits_row::Vector{Float32}, n::Int)
    indices = sortperm(logits_row, rev=true)[1:n]
    scores = logits_row[indices]
    return collect(zip(indices, scores))
end

mutable struct EntropyTracker
    history::Vector{Float32}
    trigger_positions::Vector{Int}
end

EntropyTracker() = EntropyTracker(Float32[], Int[])

function record!(tracker::EntropyTracker, e::Real, position::Int)
    push!(tracker.history, Float32(e))
    if is_uncertain(e)
        push!(tracker.trigger_positions, position)
    end
end
