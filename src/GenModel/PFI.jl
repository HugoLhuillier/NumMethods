module GenPFI

using Interpolations, ForwardDiff, NLsolve
using QuantEcon
using Logging
Logging.configure(level=INFO)

#==========================#
# IDEAS
#==========================#
# 1. Write one general function for the young, one for the old, with age as an argument
# 2. Write a general function for the expectation -- that's gonna be used in the FOC

warn("This code is still work in progress. The young2! function has not been tested yet.
Running the entire algorithm is not guaranteed to work")

#==========================#
# Types
#==========================#

"""
```
Param(R=1.01, α=0.8, β=0.98, ρ=0.8, lengthInc=4, size=20, ut="Log", ɣ=1., UI=2.)
```

Constructor of the `Param` immutable. Output the `Param` immutable with fields

- `R::Float64`: Interest rate
- `α::Float64`: Altruism parameter
- `β:Float64`: Discount rate
- `П:Array{Float64}` : Transition matrix of the income Markov chain
- `Y::Array{Float64}` : Support of the income Markov chain
- `a_grid::LinSpace` : Asset grid
- `b_grid::LinSpace` : Transfer grid
- `ut::String` : Utility function. Needs to be one of Log, CRRA, Exp or Quad
- `ɣ::Float64` : Risk aversion parameter
"""
immutable Param
  R::Float64
  α::Float64
  β::Float64
  П::Array{Float64}
  Y::Array{Float64}
  a_grid::LinSpace
  b_grid::LinSpace
  ut::String
  ɣ::Float64

  function Param(;R           = 1.01,
                  α           = 0.8,
                  β           = 0.98,
                  ρ           = 0.8,
                  lengthInc   = 4,
                  size        = 20,
                  ut          = "Log",
                  ɣ           = 1.0,
                  UI          = 2.0)

    # build the income stochastic process via Tauchen algorithm
    Y         = zeros(lengthInc, 4)
    П         = zeros(lengthInc, lengthInc)
    g(h::Int) = (h^2 - 0.4 * h^3/2)
    for h in 2:4
      mc      = QuantEcon.tauchen(lengthInc, ρ, h, g(h) - ρ/2 * g(h-1), 2)
      Y[:,h]  = mc.state_values
      if h == 4
        П[:]  = mc.p
      end
    end
    Y[:,1]    = Y[:,2] / 2
    Y[Y .< UI]= UI
    Y    = Y

    a_grid    = linspace(1e-7, 2*maximum(Y), size)
    b_grid    = linspace(1e-7, maximum(Y), size)

    utility   = ["Log"; "CRRA"; "Exp"; "Quad"]
    if !(ut in utility)
        error("Utility needs to be one Log, CRRA, Exp or Quad")
    end
    if (ut == "Log") && (ɣ != 1.0)
        error("Log utility requires ɣ = 1.0")
    end

    return new(R, α, β, П, Y, a_grid, b_grid, ut, ɣ)
  end
end

"""
```
Policies(p::Param)
```

Constructor of the Policies object, storing the policy functions. In addition, it
initializes the policy functions.

#### State spaces

- Age 1: parent's asset x transfer x child's income x parent's income
- Age 2: child's asset x parent's asset x transfer x child's income x parent's income
- Age 3: parent's asset x parent's income x child's income
- Age 4: parent's asset x child's asset x parent's income x child's income

*Initial guess*:

- Savings are linearly increasing in weatlh
- Transfers = savings adjusted by the altruistic parameter
"""
type Policies
  A1::Array{Float64}
  A2::Array{Float64}
  B3::Array{Float64}
  A3::Array{Float64}
  B4::Array{Float64}
  A4::Array{Float64}

  function Policies(p::Param)
    Ys = size(p.Y)[1]
    As = length(p.a_grid)
    Bs = length(p.b_grid)

    A1 = zeros(As,Bs,Ys,Ys)
    A2 = zeros(As,As,Bs,Ys,Ys)
    A3 = zeros(As,Ys,Ys)
    B3 = zeros(As,Ys,Ys)
    A4 = zeros(As,As,Ys,Ys)
    B4 = zeros(As,As,Ys,Ys)

    for (ai,a) in enumerate(p.a_grid)
      for (bi,b) in enumerate(p.b_grid)
        A1[:,bi,:,:]    = repeat(reshape(repeat((p.Y[:,1] + b) / (1 + p.R), inner=[As]),
                                        As, Ys, 1, 1),
                                outer = [1,1,1,Ys])
        A2[ai,:,bi,:,:] = repeat(reshape(repeat((p.Y[:,2] + b + p.R * a) / (1 + p.R), inner = [As]),
                                        As,Ys,1,1),
                                outer = [1,1,1,Ys])
        A3[ai,:,:]      = reshape(repeat((p.Y[:,3] + p.R * a) / (1 + p.R), outer = [Ys]),
                                  Ys,Ys,1)
        B3[ai,:,:]      = p.α .* A3[ai,:,:]
        A4[ai,:,:,:]    = repeat(reshape(repeat((p.Y[:,4] + p.R * a) / (1 + p.R), inner = [As]),
                                          As,Ys,1),
                                outer = [1,1,Ys])
        B4[ai,:,:,:]    = p.α .* A4[ai,:,:,:]
      end
    end

    return new(A1,A2,B3,A3,B4,A4)
  end
end

"""
```
Utility(p::Param)
```
Constructor of the `Utility` immutable, which contains the first, `du(x)`,
second, `du_p(x)`, and third, `du_s(x)`, order partial derivative of the policy functions.
"""
immutable Utility
  du::Function
  du_p::Function
  du_s::Function

  function Utility(p::Param)

    if p.ut == "Log"
      du   = (x) -> 1 / x
      du_p = (x) -> - 1 / (x * x)
      du_s = (x) -> 2 / (x * x * x)
    elseif p.ut == "CRRA"
      du   = (x) -> x^(-p.ɣ)
      du_p = (x) -> -p.ɣ * x^(-1 - p.ɣ)
      du_s = (x) -> p.ɣ * (1 + p.ɣ) * x^(-2 - p.ɣ)
    elseif p.ut == "Exp"
      du   = (x) -> exp(-p.ɣ * x)
      du_p = (x) -> -p.ɣ * exp(-p.ɣ * x)
      du_s = (x) -> p.ɣ * p.ɣ * exp(-p.ɣ * x)
    else
      du   = (x) -> 1 - p.ɣ * x
      du_p = (x) -> - p.ɣ
      du_s = (x) -> 0
    end

    return new(du, du_p, du_s)
  end
end

#==========================#
# Re-usable functions
#==========================#

# interpolate a policy function along the continuous states, and include
# a lookup table for the wages.
# `pol::String` : the name policy to be interpolated
# `X::Array` : the policy function to be interpolated
function interp(p::Param, X::Array{Float64}, pol::String)
  # the continous states are
  #        for A1: two dimensions -- old's assets and old's transfers
  #        for A2: three dimensions -- young's assets, old's assets and old's transfers
  #        for A3/B3: one dimension -- old's assets
  #        for A4/B4: two dimensions -- old's and young's assets
  # NOTE: using cubic splines. see pdf / README for more information
  Ys = size(p.Y)[1]
  if pol == "A1"
    itp = interpolate(X, (BSpline(Cubic(Line())),
                          BSpline(Cubic(Line())),
                          NoInterp(),
                          NoInterp()), OnGrid())
    itp = scale(itp, p.a_grid, p.b_grid, 1:Ys, 1:Ys)
  elseif pol == "A2"
    itp = interpolate(X, (BSpline(Cubic(Line())),
                          BSpline(Cubic(Line())),
                          BSpline(Cubic(Line())),
                          NoInterp(),
                          NoInterp()), OnGrid())
    itp = scale(itp, p.a_grid, p.a_grid, p.b_grid, 1:Ys, 1:Ys)
  elseif (pol == "A3") || (pol == "B3")
    itp = interpolate(X, (BSpline(Cubic(Line())),
                          NoInterp(),
                          NoInterp()), OnGrid())
    itp = scale(itp, p.a_grid, 1:Ys, 1:Ys)
  elseif (pol == "A4") || (pol == "B4")
    itp = interpolate(X, (BSpline(Cubic(Line())),
                          BSpline(Cubic(Line())),
                          NoInterp(),
                          NoInterp()), OnGrid())
    itp = scale(itp, p.a_grid, p.a_grid, 1:Ys, 1:Ys)
  else
    error("Policies can only be A1, A2, A3, B3, A4, B4")
  end

  return itp
end

#==========================#
# PFI functions
#==========================#

function young1!(p::Param, U::Utility, pol::Policies)
  # interpolate the future and other agents markov strategies
  B4, A4 = interp(p, pol.B4, "B4"), interp(p, pol.A4, "A4")
  B3, A3 = interp(p, pol.B3, "B3"), interp(p, pol.A3, "A3")
  A2     = interp(p, pol.A2, "A2")

  # FOC, to be solved for
  function f!(x, fvec, a_fp::Float64, b::Float64, y_c::Float64, yi_c::Int, yi_p::Int)
    Eu2 = 0
    Eu3 = 0

    # compute the t = 2 expectation
    for yi_fc in 1:length(p.Y[:,2])
      for yi_fp in 1:length(p.Y[:,4])
        _a4  = A4[a_fp, x[1], yi_fp, yi_fc]
        _b4  = B4[a_fp, x[1], yi_fp, yi_fc]
        _a2  = A2[x[1], _a4, _b4, yi_fc, yi_fp]

        # NOTE: gradient() do not access type ForwardDiff.Dual
        if typeof(x[1]) <: ForwardDiff.Dual{1,Float64}
          _da  = gradient(A4, a_fp, x[1].value, yi_fp, yi_fc)[2]
          _db  = gradient(B4, a_fp, x[1].value, yi_fp, yi_fc)[2]
        else
          _da  = gradient(A4, a_fp, x[1], yi_fp, yi_fc)[2]
          _db  = gradient(B4, a_fp, x[1], yi_fp, yi_fc)[2]
        end

        Eu2 += p.П[yi_c, yi_fc] * p.П[yi_p, yi_fp] *
              U.du(p.R * x[1] + p.Y[yi_fc, 2] + _b4 - _a2) * (p.R + _db)

        # compute the t = 3 expectation
        for yi_ffp in 1:length(p.Y[:,3])
          for yi_ffc in 1:length(p.Y[:,1])
            _a3 = A3[_a2 + _a4,yi_ffp,yi_ffc]
            _b3 = B3[_a2 + _a4,yi_ffp,yi_ffc]
            Eu3 += p.П[yi_c, yi_fc] * p.П[yi_fc, yi_ffp] * p.П[yi_fc, yi_ffc] * p.П[yi_p, yi_fp] *
                  U.du(p.R * (_a2 + _a4) + p.Y[yi_ffp,3] - _a3 - _b3)
          end
        end
      end
    end

    # ensures that one cannot consume more than what one's have
    # less costly than putting an upper bound on the solver
    if x[1] < y_c + b
      fvec[1] = U.du(y_c + b - x[1]) - p.R * p.β * (Eu2 + p.R * p.β * Eu3)
    else
      fvec[1] = 1e9
    end
  end

  function solver(a_fp::Float64, b::Float64, y_c::Float64, yi_c::Int, yi_p::Int)
    # find the roots of the FOC
    # convergence should be fast: more than 50 iterations = will never converge
    r = NLsolve.mcpsolve((x, fvec) -> f!(x, fvec, a_fp, b, y_c, yi_c, yi_p), [0.], [Inf],
                 [(b + y_c)/(1+p.R)], reformulation = :smooth, iterations = 150, autodiff = true)
    if !converged(r)
      warn("Not converged the first time: ($a_fp, $b, $y_c, $yi_c, $yi_p)")
      # trying the convergence by above
      r = NLsolve.mcpsolve((x, fvec) -> f!(x, fvec, a_fp, b, y_c, yi_c, yi_p), [0.], [Inf],
                    [b + y_c - 1e-2], reformulation = :smooth, iterations = 100, autodiff = true)
      if !converged(r)
        warn("Not converged the second time: ($a_fp, $b, $y_c, $yi_c, $yi_p)")
      end
    end
    return r.zero[1]
  end

  # i = 1
  for (ai_fp, a_fp) in enumerate(p.a_grid)
    for (bi, b) in enumerate(p.b_grid)
      for (yi_c, y_c) in enumerate(p.Y[:,1])
        for (yi_p, y_p) in enumerate(p.Y[:,3])
          # println("@ $i => ($a_fp, $b, $y_c, $yi_c, $yi_p)")
          # @time _A1 = solver(a_fp, b, y_c, yi_c, yi_p)
          _A1 = solver(a_fp, b, y_c, yi_c, yi_p)
          # _A1 may be negative due to the cubic interpolation
          pol.A1[ai_fp, bi, yi_c, yi_p] = _A1 >= 0 ? _A1 : 0.
          # i += 1
        end
      end
    end
  end
  info(" Young1! ended")

end

# TODO: to be tested
function young2!(p::Param, U::Utility, pol::Policies)
  A3, B3  = interp(p, pol.A3, "A3"), interp(p, pol.B3, "B3")

  function f!(x, fvec, a_c::Float64, a_fp::Float64, b::Float64, y_c::Float64, yi_c::Int)
    Eu3 = 0
    # compute the t = 2 expectation
    for yi_fp in 1:length(p.Y[:,3])
      for yi_fc in 1:length(p.Y[:,1])
        _b3 = B3[x[1] + a_fp,yi_fp,yi_fc]
        _a3 = A3[x[1] + a_fp,yi_fp,yi_fc]
        Eu3 += U.du(p.R * (x[1] + a_fp) + p.Y[yi_fp,3] - _b3 - a_3) * p.П[yi_c,yi_fp] * p.П[yi_c,yi_fc]
      end
    end
    # ensures that one cannot consume more than what one's have
    if x[1] < p.R * a_c + b + y_c
      fvec[1] = U.du(p.R * a_c + b + y_c - x[1]) - p.R * p.β * Eu3
    else
      fvec[1] = 1e9
    end
  end

  function solver(a_c::Float64, a_fp::Float64, b::Float64, y_c::Float64, yi_c::Int)
    # find the root of the FOC
    r = NLsolve.mcpsolve((x, fvec) -> f!(x, fvec, a_c, a_fp, b, y_c, yi_c), [0.], [Inf],
                 [(p.R * a_c + b + y_c) / (1 + p.R)], reformulation = :smooth,
                 iterations = 150, autodiff = true)
    if !converged(r)
      warn("Not converged the first time: ($a_fp, $b, $y_c, $yi_c, $yi_p)")
      # trying the convergence by above
      r = NLsolve.mcpsolve((x, fvec) -> f!(x, fvec, a_c, a_fp, b, y_c, yi_c), [0.], [Inf],
                    [b + y_c - 1e-2], reformulation = :smooth, iterations = 100, autodiff = true)
      if !converged(r)
        warn("Not converged the second time")
      end
    end
    return r.zero[1]
  end

  err = 0
  for (ai_c, a_c) in enumerate(p.a_grid)
    for (ai_fp, a_fp) in enumerate(p.a_grid)
      for (bi, b) in enumerate(b.grid)
        for (yi_c, y_c) in enumerate(p.Y[:,2])
          for yi_p in 1:length(p.Y[:,4])
            _A2 = solver(a_c,a_fp,b,y_c,yi_c)
            _A2 = _A2 >= 0. ? _A2 : 0.
            # update the distance between the two policy functions
            err = abs(pol.A2[ai_c,ai_fp,bi,yi_c,yi_p] - _A2) > err ?
                  abs(pol.A2[ai_c,ai_fp,bi,yi_c,yi_p] - _A2) : err
            pol.A2[ai_c,ai_fp,bi,yi_c,yi_p] = _A2
          end
        end
      end
    end
  end
  info(" Young2! ended")

  return err
end

function old3!(p::Param, U::Utility, pol::Policies)
  # markov strategies to be interpolated
  A4, B4  = interp(p, pol.A4, "A4"), interp(p, pol.B4, "B4")
  A1      = interp(p, pol.A1, "A1")
  A2      = interp(p, pol.A2, "A2")

  function f!(x, fvec, a_p::Float64, y_p::Float64, y_c::Float64, yi_p::Int, yi_c::Int)
    _a1          = A1[x[1], x[2], yi_c, yi_p]
    if typeof(x[1]) <: ForwardDiff.Dual{2,Float64}
      _da_a, _da_b = gradient(A1, x[1].value, x[2].value, yi_c, yi_p)
    else
      _da_a, _da_b = gradient(A1, x[1], x[2], yi_c, yi_p)
    end

    # compute the t = 4 expectations
    Eu4, Eu2 = 0., 0.
    for yi_fp in 1:length(p.Y[:,4])
      for yi_fc in 1:length(p.Y[:,2])
        _a4  = A4[x[1],_a1,yi_fp,yi_fc]
        _b4  = B4[x[1],_a1,yi_fp,yi_fc]
        _a2  = A2[_a1,_a4,_b4,yi_fc,yi_fp]

        Eu4 += U.du(p.R * x[1] + p.Y[yi_fp,4] - _b4 - _a4) * p.П[yi_p,yi_fp] * p.П[yi_c,yi_fc]
        Eu2 += U.du(p.R * _a1 + p.Y[yi_fc,2] + _b4 - _a2) * p.П[yi_p,yi_fp] * p.П[yi_c,yi_fc]
      end
    end

    # ensures that one cannot consume more than what one's have
    if x[1] + x[2] < p.R * a_p + y_p
      fvec[1] = U.du(p.R * a_p + y_p - x[2] - x[1]) + p.α * _da_a * U.du(y_c + x[2] - _a1) -
              p.R * p.β * (Eu4 + p.α * _da_a * Eu2)
      fvec[2] = U.du(p.R * a_p + y_p - x[2] - x[1]) - p.α * (1 - _da_b) * U.du(y_c + x[2] - _a1) -
              p.α * p.R * p.β * _da_b * Eu2
    else
      fvec[1] = 1e9
      fvec[2] = 1e9
    end
  end

  function solver(a_p::Float64, y_p::Float64, y_c::Float64, yi_p::Int, yi_c::Int, ai_p::Int)
    # find the roots of the system of equations. try to guess in a smart way the
    # initial value
    c0 = (p.R * a_p + y_p) / 7
    a0 = min(p.R * a_p + y_p - c0, y_p + a_p / (1 + p.R))
    b0 = max(0., p.R * a_p + y_p - c0 - a0)
    r = NLsolve.mcpsolve((x, fvec) -> f!(x, fvec, a_p, y_p, y_c, yi_p, yi_c), [0., 0.], [Inf, Inf],
                 [a0, b0], reformulation = :smooth, iterations = 150, autodiff = true)
    if !converged(r)
      # warn("Did not converged the first time: ($a_p, $y_p, $y_c, $yi_p, $yi_c)")
      # in most of the cases, starting with the previous results, and using nlsolve instead
      # mcpsolve works. NOTE: because the previous model has not converged,
      # we can be sure that the solution is not a corner solution
      r = NLsolve.nlsolve((x, fvec) -> f!(x, fvec, a_p, y_p, y_c, yi_p, yi_c),
                   r.zero, iterations = 200)
      if !converged(r)
        # warn("Did not converged the second time: ($a_p, $y_p, $y_c, $yi_p, $yi_c)")
        _A3, _B3 = pol.A3[ai_p,yi_p,yi_c], pol.B3[ai_p,yi_p,yi_c]
      else
        _A3, _B3 = r.zero
      end
    else
      _A3, _B3 = r.zero
    end
    return (_A3, _B3)
  end

  for (ai_p, a_p) in enumerate(p.a_grid)
    for (yi_p, y_p) in enumerate(p.Y[:,3])
      for (yi_c,y_c) in enumerate(p.Y[:,1])
        _A3, _B3 = solver(a_p, y_p, y_c, yi_p, yi_c, ai_p)
        pol.A3[ai_p,yi_p,yi_c] = _A3 >= 0. ? _A3 : 0.
        pol.B3[ai_p,yi_p,yi_c] = _B3 >= 0. ? _B3 : 0.
      end
    end
  end
  info(" Old3! ended")
end

function old4!(p::Param, U::Utility, pol::Policies)
  # three markov strategies need to be interpolated: A2, A3 and B3
  A2, A3, B3 = interp(p, pol.A2, "A2"), interp(p, pol.A3, "A3"), interp(p, pol.B3, "B3")

  function f!(x, fvec, a_p::Float64, y_p::Float64, a_c::Float64, yi_c::Int, yi_p::Int)
    Euc = 0
    _a2  = A2[a_c, x[1], x[2], yi_c, yi_p]
    # compute the t = 3 expectation
    for yi_fp in 1:length(p.Y[:,3])
      for yi_fc in 1:length(p.Y[:,1])
        _b3  = B3[_a2 + x[1],yi_fp,yi_fc]
        _a3  = A3[_a2 + x[1],yi_fp,yi_fc]
        Euc += U.du(p.R * (_a2 + x[1]) + p.Y[yi_fp,3]-_b3-_a3) * p.П[yi_c, yi_fp] * p.П[yi_c, yi_fc]
      end
    end

    # ensures that one cannot consume more than what one's have
    if x[1] + x[2] < p.R + a_p + y_p
      fvec[1] = U.du(p.R * a_p + y_p - x[2] - x[1]) - p.R * p.β * p.α * Euc
      fvec[2] = U.du(p.R * a_p + y_p - x[2] - x[1]) - p.α * U.du(p.R * a_c + p.Y[yi_c,2]+x[2]-_a2)
    else
      fvec[1] = 1e9
      fvec[2] = 1e9
    end
  end

  function solver(a_p::Float64, y_p::Float64, a_c::Float64, yi_c::Int, yi_p::Int)
    r = NLsolve.mcpsolve((x, fvec) -> f!(x, fvec, a_p, y_p, a_c, yi_c, yi_p), [0., 0.], [Inf, Inf],
                 [0., 0.], reformulation = :smooth, autodiff = true, iterations = 150)
    if !converged(r)
      # use the same procedure as for the old3! function. in the first iteration, always converges in
      # with the first mcpsolve
      r = NLsolve.nlsolve((x, fvec) -> f!(x, fvec, a_p, y_p, a_c, yi_c, yi_p),
                   r.zero, iterations = 150)
      if !converged(r)
        warn("Not converged the second time")
      end
    end
    return r.zero
  end

  for (ai_p, a_p) in enumerate(p.a_grid)
    for (ai_c, a_c) in enumerate(p.a_grid)
      for (yi_p, y_p) in enumerate(p.Y[:,4])
        for yi_c in 1:length(p.Y[:,2])
          _A4, _B4 = solver(a_p, y_p, a_c, yi_c, yi_p)
          # Both may be negative due to approximations
          pol.A4[ai_p, ai_c, yi_p, yi_c] = _A4 >= 0. ? _A4 : 0.
          pol.B4[ai_p, ai_c, yi_p, yi_c] = _B4 >= 0. ? _B4 : 0.
        end
      end
    end
  end
  info(" Old4! ended")
end
#
# function ite(maxIter::Int, p::Param;
#              tol::Float64=1e-8, isCheck::Bool=false)#::Union{Policies,Base.#error}
#
#   U   = Utility(p)
#   pol = Policies(p)
#
#   for iter in 1:maxIter
#     old4!(p, U, pol)
#     young1!(p, U, pol)
#     old3!(p, U, pol)
#     err = young2!(p, U, pol)
#
#     if isCheck
#       println("Error @ iteration $iter: ", err)
#     end
#     if err < tol
#       println("Find solutions after $iter iterations")
#       return pol
#       break
#     elseif iter == maxIter
#       error("No solutions found after $iter iterations")
#     end
#
#   end
#
# end


end
