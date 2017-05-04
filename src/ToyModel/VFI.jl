module ToyVFI

using Interpolations, Optim
export getSol, ToyModel, ite

#==========================#
# Types
#==========================#

"""
Type storing the information of the 2-period simplified game

##### Fields

- `R::Float64` : The interest rate plus 1 (strictly greater than 1)
- `beta::Float64` : Discount rate in (0, 1)
- `alpha:Float64` : Altruistic factor in (0, 1)
- `Pi::Matrix{Floa64}` : Transition matrix for newborn income
- `inc_vals::Vector{Float64}` : Levels of income
- `transfer_grid::LinSpace{Float64}` : Transfer grid
- `asset_grid::LinSpace{Float64}` : Asset grid
- `utility::String` : Utility function. Can choose between "Log", "CRRA", "Exp" and "Quad"
- `gamma::Float64` : Coefficient of risk aversion.
"""
type ToyModel
  R::Float64
  beta::Float64
  alpha::Float64
  Pi::Array{Float64}
  inc_vals::Array{Float64}
  b_grid::AbstractArray
  a_grid::AbstractArray
  utility::String
  gamma::Float64
end

function ToyModel(;R        = 1.01,
                  beta      = 0.98,
                  alpha     = 0.8,
                  Pi        =  [0. 1.;
                                1. 0.],
                  inc_vals  = [1., 6.],
                  max_a     = 2 * maximum(inc_vals),
                  max_b     = maximum(inc_vals),
                  grid_size = 100,
                  ut        = "Log",
                  gamma     = 1.0)

    lb_a         = inc_vals[1] / (alpha * R)

    a_grid       = linspace(1e-7, max_a, grid_size)
    b_grid       = linspace(1e-7, max_b, grid_size)

    utility          = ["Log"; "CRRA"; "Exp"; "Quad"]
    if !(ut in utility)
        error("Utility needs to be of the form Log, CRRA, Exp or Quad")
    end
    if (ut == "Log") && (gamma != 1.0)
        error("Log utility requires gamma = 1")
    end

    return ToyModel(R, beta, alpha, Pi, inc_vals, b_grid, a_grid, ut, gamma)
end

#==========================#
# Re-usable functions
#==========================#

function pick_u(mod::ToyModel)
  gamma, ut      = mod.gamma, mod.utility

  if ut == "Log"
    du = (x) -> log(x)
  elseif ut == "CRAA"
    du = (x) -> (x^(1-gamma) - 1) / (1 - gamma)
  elseif ut == "Exp"
    du = (x) -> (1 - exp(-gamma * x)) / gamma
  else
    du = (x) -> x - gamma * x^2 / 2
  end

  return du
end

function itp_pol(mod::ToyModel, cha::Char, x::AbstractMatrix)

  siz      = size(x)
  itp      = interpolate(x, (BSpline(Linear()), NoInterp()), OnGrid())
  if cha == 'p'
    # Interpolate the asset policy function, therefore along the transfer grid
    itp_s  = scale(itp, mod.b_grid, 1:siz[2])
  elseif cha == 'v'
    # Interpolate the old value function along the asset grid
    itp_s  = scale(itp, mod.a_grid, 1:siz[2])
  end

  return itp_s
end

#==========================#
# VFI functions
#==========================#

function young!(_VO::Matrix, _A::Matrix, mod::ToyModel)
  VO = itp_pol(mod, 'v', _VO)
  u  = pick_u(mod)

  function obj_wrap(y::Float64, yi::Integer, b::Float64)
    f = function(x::Float64)
      VOp = 0
      for (_yi, _y) in enumerate(mod.inc_vals)
        VOp += mod.Pi[yi, _yi] * VO[x, _yi]
      end
      return - u(y + b - x) - mod.beta * VOp
    end
    return f
  end

  function max_solver(y::Float64, yi::Integer, b::Float64)
    f = obj_wrap(y, yi, b)
    return optimize(f, 0., y + b).minimizer
  end

  for (yi, y) in enumerate(mod.inc_vals)
    for (bi, b) in enumerate(mod.b_grid)
      _A[bi, yi] = max_solver(y, yi, b)
    end
  end
end

function old!(_VO::Matrix, _A::Matrix, _B::Matrix, mod::ToyModel)
  A    = itp_pol(mod, 'p', _A)
  VO_1 = zeros(size(_VO))
  u    = pick_u(mod)

  function obj_wrap(y::Float64, yi::Integer, a::Float64)
    f = function(x::Float64)
      if y + x - A[x, yi] > 0
        return - u(mod.R * a - x) - mod.alpha * u(y + x - A[x, yi])
      else
        return Inf
      end
    end
    return f
  end

  function max_solver(y::Float64, yi::Integer, a::Float64)
    f = obj_wrap(y, yi, a)
    return optimize(f, 0., mod.R * a).minimizer
  end

  for (yi, y) in enumerate(mod.inc_vals)
    for (ai, a) in enumerate(mod.a_grid)
      _B[ai, yi]   = max_solver(y, yi, a)
      if y + _B[ai, yi] - A[_B[ai, yi], yi] < 0
        println("Here, error")
      end
      VO_1[ai, yi] = u(mod.R * a - _B[ai, yi]) +
                    mod.alpha * u(y + _B[ai, yi] - A[_B[ai, yi], yi])
    end
  end

  return VO_1
end

function init_values(mod::ToyModel)
  VO = randn(length(mod.a_grid), length(mod.inc_vals))
  A  = zeros(length(mod.b_grid), length(mod.inc_vals))
  B  = zeros(length(mod.a_grid), length(mod.inc_vals))

  return VO, A, B
end

function ite(maxIter::Integer, mod::ToyModel;
             tol::Float64 = 1e-6, isCheck::Bool = false)

  VO_0, A, B = init_values(mod)
  for iter in 1:maxIter
    young!(VO_0, A, mod)
    VO_1 = old!(VO_0, A, B, mod)
    err  = maxabs(VO_1 - VO_0)

    if err < tol
      println("Find the the solutions after $iter iterations")
      return A, B
      break
    elseif iter == maxIter
      println("Did not find the solutions after $iter iterations.")
    end

    if isCheck
      println("Iteration #$iter: error = ", err)
    end

    VO_0 = copy(VO_1)
  end
end

#==========================#
# Solution VFI
#==========================#

"""
```
getSol(;R=1.01, beta=0.98, alpha=0.8, Pi=[0. 1.;1. 0.], inc_vals = [1., 6.], grid_size = 500, ut = "Log", gamma = 1.)
```

Solves the toy model for the parameters passed using value function iteration.

Returns an array of tuple containing the policy functions.
1. Savings. Columns = income value
1. Transfer. Columns = income value of the child
"""
function getSol(;R = 1.01,
                beta = 0.98,
                alpha = 0.8,
                Pi = [0. 1.; 1. 0.],
                inc_vals = [1., 6.],
                grid_size = 100,
                ut = "Log",
                gamma = 1.0)


  olg = ToyModel(R = R, beta = beta, alpha = alpha, Pi = Pi, inc_vals = inc_vals,
                grid_size = grid_size, ut = ut, gamma = gamma)
  sol = ite(100,olg)
  return sol
end

end
