using PackageCompiler

create_sysimage(
    [:PythonCall, :Plots, :Statistics, :JSON3, :HTTP, :Random],
    sysimage_path="witty_sysimage.so",
    precompile_execution_file="index.jl"
)

println("✅ Sysimage built! Now run:")
println("   julia --sysimage witty_sysimage.so index.jl")