using Plots

function plot_entropy_profile(tracker::EntropyTracker,
    actual_length::Int,
    threshold::Float32=Float32(ENTROPY_THRESHOLD),
    title_str::String="Entropy Profile")

    history = tracker.history[1:actual_length]
    positions = 1:actual_length

    p = plot(
        positions,
        history,
        label="Entropy",
        linewidth=2,
        color=:blue,
        marker=:circle,
        markersize=4,
        title=title_str,
        xlabel="Token Position",
        ylabel="Entropy",
        legend=:topright,
        ylims=(0, max(maximum(history) * 1.1, threshold * 1.2))
    )

    hline!(p, [threshold],
        label="Threshold ($(threshold))",
        color=:red,
        linestyle=:dash,
        linewidth=2
    )

    for pos in tracker.trigger_positions
        if pos <= actual_length
            scatter!(p, [pos], [history[pos]],
                color=:red,
                markersize=10,
                markershape=:star5,
                label="Uncertain (pos $pos)"
            )
        end
    end

    return p
end

function plot_charades_decision(candidates::Vector{CandidateResult},
    winner::CandidateResult)

    names = [r.token_text for r in candidates]
    entropies = [r.avg_entropy for r in candidates]
    coherences = [r.coherence_score for r in candidates]

    colors = [r.token_text == winner.token_text ? :green : :gray for r in candidates]

    p1 = bar(names, entropies,
        title="Avg Entropy by Candidate\n(lower = better)",
        ylabel="Avg Entropy",
        color=colors,
        legend=false,
        xlabel="Candidate Token"
    )

    p2 = bar(names, coherences,
        title="Coherence Score by Candidate\n(higher = better)",
        ylabel="Coherence Score",
        color=colors,
        legend=false,
        xlabel="Candidate Token"
    )

    return plot(p1, p2, layout=(1, 2), size=(800, 400))
end

function plot_profiler(profiler::Profiler)
    if isempty(profiler.decisions)
        println("No decisions logged yet!")
        return
    end

    slopes = [d.slope for d in profiler.decisions]
    labels = ["'$(d.token_text)'" for d in profiler.decisions]
    colors = [d.was_good ? :green : :red for d in profiler.decisions]

    p1 = bar(labels, slopes,
        title="Entropy Slope per Decision\n(negative = good)",
        ylabel="Slope",
        color=colors,
        legend=false,
        xlabel="Decision"
    )

    running_accuracy = Float32[]
    good_count = 0
    for (i, d) in enumerate(profiler.decisions)
        if d.was_good
            good_count += 1
        end
        push!(running_accuracy, good_count / i)
    end

    p2 = plot(1:length(running_accuracy), running_accuracy,
        title="Running Accuracy",
        ylabel="Accuracy",
        xlabel="Decision #",
        linewidth=2,
        color=:blue,
        ylims=(0, 1.1),
        label="Accuracy",
        legend=false
    )
    hline!(p2, [0.5], color=:red, linestyle=:dash, label="50% baseline")

    return plot(p1, p2, layout=(1, 2), size=(800, 400))
end

function save_all_plots(tracker::EntropyTracker,
    actual_length::Int,
    candidates::Vector{CandidateResult},
    winner::CandidateResult,
    profiler::Profiler)

    println("\n📈 Generating visualizations...")

    p1 = plot_entropy_profile(tracker, actual_length)
    savefig(p1, "entropy_profile.png")
    println("✅ Saved entropy_profile.png")

    p2 = plot_charades_decision(candidates, winner)
    savefig(p2, "charades_decision.png")
    println("✅ Saved charades_decision.png")

    p3 = plot_profiler(profiler)
    savefig(p3, "profiler_results.png")
    println("✅ Saved profiler_results.png")

    println("Done! Open the PNG files to view results.")
end