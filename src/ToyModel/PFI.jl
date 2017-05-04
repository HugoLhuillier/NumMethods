module ToyPFI

using Roots, Interpolations
using Plots, LaTeXStrings

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
  transfer_grid::AbstractArray
  asset_grid::AbstractArray
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

    lb_a               = inc_vals[1] / (alpha * R)

    asset_grid       = linspace(1e-7, max_a, grid_size)
    transfer_grid    = linspace(1e-7, max_b, grid_size)

    utility          = ["Log"; "CRRA"; "Exp"; "Quad"]
    if !(ut in utility)
        error("Utility needs to be of the form Log, CRRA, Exp or Quad")
    end
    if (ut == "Log") && (gamma != 1.0)
        error("Log utility requires gamma = 1")
    end

    return ToyModel(R, beta, alpha, Pi, inc_vals, transfer_grid, asset_grid, ut, gamma)
end

#==========================#
# Re-usable functions
#==========================#

# returns the derivative of the utility function
function pick_du(olg::ToyModel)
  gamma, ut      = olg.gamma, olg.utility

  if ut == "Log"
    du = (x) -> 1 / x
  elseif ut == "CRRA"
    du = (x) -> x^(-gamma)
  elseif ut == "Exp"
    du = (x) -> Exp(-gamma * x)
  elseif ut == "Quad"
    du = (x) -> 1 - gamma * x
  end

  return du
end

# interpolate a policy function along one dimension (asset or transfer), and include
# a lookup table for the wage.
# `pol::Char` : `a` if want to interpolate the asset policy function, `b` for the transfer
# `x::AbstractMatrix` : Matrix of the policy function to be interpolated
function itp_pol(olg::ToyModel, pol::Char, x::AbstractMatrix)

  siz      = size(x)
  itp      = interpolate(x, (BSpline(Linear()), NoInterp()), OnGrid())
  if pol == 'a'
    # Interpolate the asset policy function, therefore along the transfer grid
    itp_s  = scale(itp, olg.transfer_grid, 1:siz[2])
  else
    # Interpolate the transfer policy function, therefore along the asset grid
    itp_s  = scale(itp, olg.asset_grid, 1:siz[2])
  end

  return itp_s
end

#==========================#
# PFI functions
#==========================#

function young!(_B_c::Matrix,
                _A_c::Matrix,
                olg::ToyModel)

    R, beta, Pi    = olg.R, olg.beta, olg.Pi
    inc, b_g       = olg.inc_vals, olg.transfer_grid
    du             = pick_du(olg)

    # Interpolate the transfer policy function & return FOC
    function foc_wrap(_y::Float64, _y_idx::Integer, _b::Float64)
      B_c = itp_pol(olg, 'b', _B_c)
      foc = function(_x)
        e_du = 0
        for (yc_idx, yc) in enumerate(inc)
          if B_c[_x, yc_idx] > R * _x
            bb    = R * _x
          else
            bb    = B_c[_x, yc_idx]
          end
          if Pi[_y_idx, yc_idx] > 0
            e_du += du(R * _x - bb) * Pi[_y_idx, yc_idx]
          end
        end
        return du(_y + _b - _x) - beta * R * e_du
      end

      return foc
    end

    for (y_idx, y) in enumerate(inc)
      for (b_idx, b) in enumerate(b_g)

        # If currently poor, then kid will be rich. And vice versa
        foc   = foc_wrap(y, y_idx, b)
        if foc(0.) > 0
            # corner solution
            _A_c[b_idx, y_idx] = 0
        else
            # interior solution
            a_star             = fzero(foc, 0., y + b)
            _A_c[b_idx, y_idx] = a_star
        end
      end
    end
end

function old(_B_c::Matrix,
              _A_c::Matrix,
              olg::ToyModel)

  alpha, R       = olg.alpha, olg.R
  inc, a_g       = olg.inc_vals, olg.asset_grid
  du             = pick_du(olg)
  B_c            = zeros(size(_B_c))

  # return the FOC as a function of the choice
  function foc_wrap(_a::Float64, _y::Float64, _yi::Integer)
    A_c       = itp_pol(olg, 'a', _A_c)
    da = (_x) -> gradient(A_c, _x, _yi)
    foc       = function(_x::Float64)
      if A_c[_x, _yi] > _y + _x
        aa = _y + _x
      else
        aa = A_c[_x, _yi]
      end
      return du(R * _a - _x) - alpha * (1 - da(_x)[1]) * du(_y + _x - aa)
    end
    return foc
  end

  # solve for the optimal choice
  function foc_solver(_a::Float64, _y::Float64, _yi::Integer)
    foc   = foc_wrap(_a, _y, _yi)
    if foc(0.) > 0
      b_star = 0
    else
      b_star = fzero(foc, 0, R * _a)
    end
    return b_star
  end

  for (y_idx, y) in enumerate(inc)
    for (a_idx, a) in enumerate(a_g)
      B_c[a_idx, y_idx] = foc_solver(a, y, y_idx)
    end
  end

  return B_c
end

function init_values(olg::ToyModel, isRand::Bool)
  b_g, a_g, inc, R   = olg.transfer_grid, olg.asset_grid, olg.inc_vals, olg.R

  if isRand
    B_c         = randn(length(a_g), length(inc))
    A_c         = randn(length(b_g), length(inc))
  else
    B_c       = zeros(length(a_g), length(inc))
    A_c       = zeros(length(b_g), length(inc))
    for (y_idx, y) in enumerate(inc)
      if y_idx == 1
        # Poor kid <-> rich parent
        B_c[:,y_idx] = a_g ./ 2
        A_c[:,y_idx] = (y .+ b_g) ./ (1 + R)
      else
        # Rich kid <-> poor parent
        B_c[:, y_idx] = 0.
        A_c[:, y_idx] = (y .+ b_g) ./ (1 + R)
      end
    end
  end
  return A_c, B_c
end

function ite(max_iter::Integer,
            olg::ToyModel;
            tol::Float64=1e-7,
            check::Bool=false,
            plotiter::Bool=false)

    A_c, B_c = init_values(olg,plotiter)
    if plotiter
      Aplot       = zeros(length(olg.transfer_grid), 5)
      Bplot       = zeros(length(olg.asset_grid), 5)
      labels      = Array{String}(5)
      Aplot[:,1]  = A_c[:,1]
      Bplot[:,1]  = B_c[:,1]
      labels[1]   = "Init."
    end
    for iter in 1:max_iter
      # update guess for young and old policy function, in that order
      young!(B_c, A_c, olg)
      B_cn = old(B_c, A_c, olg)
      err  = maxabs(B_cn - B_c)

      if plotiter && (iter <= 3)
        Aplot[:,iter+1] = A_c[:,1]
        Bplot[:,iter+1] = B_cn[:,1]
        labels[iter+1]  = ""
      end

      # check convergence
      if err < tol
        println("Found solution after $iter iterations")

        if plotiter
          Aplot[:,5] = A_c[:,1]
          Bplot[:,5] = B_cn[:,1]
          labels[5]  = "P.$iter"
          C(g::ColorGradient) = RGB[g[z] for z=linspace(0,1,5)]
          g             = :amp
          colors        = C(cgrad(g))

          plot_1 = Plots.plot(olg.transfer_grid, Aplot, grid = false,
                              label = labels', color = colors', title = "Savings",
                              xlab = L"b", ylab = L"\mathcal{A}(b, \, y)")
          plot_2 = Plots.plot(olg.asset_grid, Bplot, grid = false,
                              label = labels', color = colors', title = "Transfer",
                              xlab = L"a", ylab = L"\mathcal{B}(a, \, y)")
          display(Plots.plot(plot_1, plot_2))
        end

        cy, co = zeros(size(A_c)), zeros(size(B_cn))
        for (j, y) in enumerate(olg.inc_vals)
          cy[:,j] = y + collect(olg.transfer_grid) - A_c[:,j]
          co[:,j] = olg.R * collect(olg.asset_grid) - B_cn[:,j]
        end

        return A_c, B_cn, cy, co
        break
      elseif iter == max_iter
          error("No solution found after $iter iterations")
      end
      if check
        println("Iteration # $iter. Error: ", err)
      end
      # update guess
      B_c = copy(B_cn)
    end
end

#==========================#
# Solution PFI
#==========================#

"""
```
getSol(;R=1.01, beta=0.98, alpha=0.8, Pi=[0. 1.;1. 0.], inc_vals = [1., 6.], grid_size = 100, ut = "Log", gamma = 1.)
```

Solves the toy model for the parameters passed using policy function iteration.

Returns an array of tuple containing the policy functions.
1. Savings. Columns = income value
1. Transfer. Columns = income value of the child
1. Young consumption. Columns = income value
1. Old consumption. Columns = income value of the child
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
  sol = ite(30,olg)
  return sol
end

end
