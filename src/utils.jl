module utils
using CUDA, LinearAlgebra, GilaElectromagnetics, JLD2
export fun_to_mat, load_greens_operator


function fun_to_mat(A::Function, n::Int)
    result = CUDA.zeros(ComplexF64, n, n)
    for j in 1:n
        col = zeros(ComplexF64, n)
        col[j] = 1.0
        result[:, j] = A(CuArray(col))
    end
    return result
end
Base.:\(f::Function, x::AbstractArray) = fun_to_mat(f, size(x, 1)) \ x


function load_greens_operator(cells::NTuple{3, Int}, scale::NTuple{3, Rational{Int}}; preload_dir="data")
    fname = "$(cells[1])x$(cells[2])x$(cells[3])_$(scale[1].num)ss$(scale[1].den)x$(scale[2].num)ss$(scale[2].den)x$(scale[3].num)ss$(scale[3].den).jld2"
    fpath = joinpath(preload_dir, fname)
    if isfile(fpath)
            file = jldopen(fpath)
            fourier = CuArray.(file["fourier"])
            options = GlaKerOpt(true)
            volume = GlaVol(cells, scale, (0//1, 0//1, 0//1))
            mem = GlaOprMem(options, volume; egoFur=fourier, setTyp=ComplexF64)
            return GlaOpr(mem)
    end
    operator = GlaOpr(cells, scale; setTyp=ComplexF64, useGpu=true)
    fourier = Array.(operator.mem.egoFur)
    jldsave(fpath; fourier=fourier)
    return operator
end
end