mutable struct GenerationState
    tokens::Vector{Int}
    entropy_tracker::EntropyTracker
    current_position::Int
    generated_text::String
end

GenerationState() = GenerationState(Int[], EntropyTracker(), 0, "")

function load_model(model_path::String)
    llama_cpp = pyimport("llama_cpp")
    return llama_cpp.Llama(
        model_path=model_path,
        n_ctx=4096,
        logits_all=true,
        embedding=true,      # dual flag test
        n_threads=4,         # generation speed
        n_threads_batch=8,   # batch speed
        verbose=false
    )
end

function generate_token_by_token(model, prompt::String, max_tokens::Real)
    state = GenerationState()
    output = model(prompt, max_tokens=max_tokens, echo=true)
    raw_logits = model.scores
    logits_matrix = copy(pyconvert(Array{Float32}, raw_logits))

    println("Logits matrix shape: $(size(logits_matrix))")
    println("Processing $(size(logits_matrix, 1)) positions...")

    py_tokens = model.tokenize(pybuiltins.bytes(prompt, "utf-8"))
    token_list = pyconvert(Vector{Int64}, py_tokens)
    actual_length = length(token_list)

    println("Prompt has $actual_length tokens, ignoring padding positions")

    println(typeof(token_list))
    println(typeof(actual_length))



    for pos in 1:actual_length
        current_logits = logits_matrix[pos, 1:end]
        e = entropy(current_logits)
        record!(state.entropy_tracker, e, pos)
        state.current_position = pos

        if is_uncertain(e)
            top = top_n_tokens(current_logits, 3)
            println(" Position $pos | entropy=$(round(e, digits=2)) | UNCERTAIN")
            println("    Top 3 candidates: $top")
        else
            println(" Position $pos | entropy=$(round(e, digits=2)) | confident")
        end
    end

    println()
    println("Total uncertain positions: $(length(state.entropy_tracker.trigger_positions))")
    println("Trigger positions: $(state.entropy_tracker.trigger_positions)")

    return state, logits_matrix
end

function reconstruct_prompt_to_position(token_list::Vector{Int64}, model, position::Int)
    tokens_so_far = token_list[1:position]
    bytes = model.detokenize(tokens_so_far)
    return pyconvert(String, bytes.decode("utf-8", errors="replace"))
end