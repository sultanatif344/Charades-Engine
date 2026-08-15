# api.jl
# Charades Engine HTTP API - OpenAI compatible proxy for SillyTavern

using HTTP
using JSON3
using Random

const API_PORT = 8080

function extract_prompt(body::JSON3.Object)
    messages = body["messages"]

    println("📨 Message count: $(length(messages))")
    println("📨 Last message role: $(messages[end]["role"])")

    # Build full conversation context
    parts = String[]
    for msg in messages
        role = msg["role"]
        content = msg["content"]
        if role == "system"
            push!(parts, "System: $content")
        elseif role == "user"
            push!(parts, content)
        elseif role == "assistant"
            push!(parts, content)
        end
    end

    return join(parts, "\n")
end

function build_response(content::String, model_name::String="witty-v4")
    return Dict(
        "id" => "chatcmpl-$(randstring(10))",
        "object" => "chat.completion",
        "model" => model_name,
        "choices" => [Dict(
            "index" => 0,
            "message" => Dict(
                "role" => "assistant",
                "content" => content
            ),
            "finish_reason" => "stop"
        )],
        "usage" => Dict(
            "prompt_tokens" => 0,
            "completion_tokens" => 0,
            "total_tokens" => 0
        )
    )
end

function handle_completion(body::JSON3.Object)
    prompt = extract_prompt(body)
    max_tokens = get(body, "max_tokens", 200)

    println("\n📨 Received prompt: '$(prompt[1:min(80, length(prompt))])...'")

    # Run entropy scan
    state, logits = generate_token_by_token(model, prompt, 1)

    py_tokens = model.tokenize(pybuiltins.bytes(prompt, "utf-8"))
    token_list = pyconvert(Vector{Int64}, py_tokens)
    actual_length = length(token_list)

    trigger_entropies = [(pos, state.entropy_tracker.history[pos])
                         for pos in state.entropy_tracker.trigger_positions]

    sort!(trigger_entropies, by=x->x[2], rev=true)

    top_triggers = [pos for (pos, _) in trigger_entropies[1:min(MAX_TRIGGERS, length(trigger_entropies))]]

    sort!(top_triggers)

    corrections_made = 0

    for pos in top_triggers
        local uncertain_logits, active_tags, winner, trajectory, candidates

        uncertain_logits = logits[pos, 1:end]
        partial_prompt = reconstruct_prompt_to_position(token_list, model, pos)
        active_tags = ask_witty_for_tags(model, prompt, hints)

        winner, trajectory, candidates = charades(
            model, partial_prompt, uncertain_logits,
            active_tags, hints, model, cache, :any
        )

        if winner !== nothing
            corrections_made += 1
            context = get_context_fingerprint(model, token_list, pos)
            log_decision!(profiler, context,
                winner.token_id, winner.token_text,
                winner.avg_entropy, winner.coherence_score, trajectory)
        end
    end

    println("✅ Corrections made: $corrections_made")

    # Generate final response from Witty
    output = model(prompt, max_tokens=max_tokens, echo=false)
    response_text = pyconvert(String, output["choices"][0]["text"])

    println("📤 Response length: $(length(response_text)) chars")
    println("📤 Response preview: '$(response_text[1:min(100, length(response_text))])'")

    # Check if all positions were bad - graceful degradation
    if corrections_made == 0 && length(top_triggers) > 0
        all_skipped = all(e -> e > 7.0,
            [state.entropy_tracker.history[p] for p in top_triggers])
        if all_skipped
            response_text = "I'm not confident I can reliably continue this. " *
                            "Could you provide more context or rephrase your prompt?"
        end
    end

    return response_text
end

function start_api()
    println("🚀 Charades Engine API starting on port $API_PORT")
    println("   Point SillyTavern to: http://0.0.0.0:$API_PORT")
    println("   OpenAI compatible endpoint: /v1/chat/completions")

    HTTP.serve("0.0.0.0", API_PORT) do request
        try
            target = String(request.target)

            # Health check
            if request.method == "GET" && target == "/health"
                return HTTP.Response(200,
                    ["Content-Type" => "application/json"],
                    JSON3.write(Dict("status" => "ok", "model" => "witty-v4")))
            end

            # Models list (SillyTavern checks this)
            if request.method == "GET" && target == "/v1/models"
                return HTTP.Response(200,
                    ["Content-Type" => "application/json"],
                    JSON3.write(Dict(
                        "data" => [Dict(
                            "id" => "witty-v4",
                            "object" => "model"
                        )]
                    )))
            end

            # Main completion endpoint
            if request.method == "POST" &&
               (target == "/v1/chat/completions" || target == "/v1/completions")

                body = JSON3.read(String(request.body))
                response_text = handle_completion(body)

                return HTTP.Response(200,
                    ["Content-Type" => "application/json"],
                    JSON3.write(build_response(response_text)))
            end

            return HTTP.Response(404, "Not found")

        catch e
            println("❌ Error: $e")
            return HTTP.Response(500,
                ["Content-Type" => "application/json"],
                JSON3.write(Dict("error" => string(e))))
        end
    end
end