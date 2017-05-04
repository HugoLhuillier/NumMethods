module ToyEGM

using Interpolations
using Plots, LaTeXStrings

export getSol, ToyModel, ite

#==========================#
# Types
#==========================#

"""
Type storing the information of the 2-period toy model.

##### Fields

- `R::Float64` : The interest rate plus 1 (strictly greater than 1)
- `beta::Float64` : Discount rate in (0, 1)
- `alpha:Float64` : Altruistic factor in (0, 1)
- `Pi::Matrix{Floa64}` : Transition matrix for newborn income
- `inc::Vector{Float64}` : Levels of income
- `b_grid::LinSpace{Float64}` : Transfer grid
- `a_grid::LinSpace{Float64}` : Asset grid
- `utility::String` : Utility function. Can choose between "Log", "CRRA", "Exp" and "Quad"
- `gamma::Float64` : Coefficient of risk aversion.
"""
type ToyModel
  R::Float64
  beta::Float64
  alpha::Float64
  Pi::Array{Float64}
  inc::Array{Float64}
  b_grid::AbstractArray
  a_grid::AbstractArray
  ut::String
  gamma::Float64
end

function ToyModel(;R        = 1.01,
                  beta      = 0.98,
                  alpha     = 0.8,
                  Pi        =  [0. 1.;
                                1. 0.],
                  inc       = [1., 6.],
                  max_a     = Void,
                  max_b     = Void,
                  grid_size = 100,
                  ut        = "Log",
                  gamma     = 1.0)

  utility = ["Log"; "CRRA"; "Exp"; "Quad"]
  if !(ut in utility)
      error("Utility needs to be one of Log, CRRA, Exp or Quad")
  end
  if (ut == "Log") && (gamma != 1.0)
      error("Log utility requires gamma = 1")
  end

  if (max_a == Void) && (max_b == Void)
    max_b  = maximum(inc)
    max_a  = 2 * max_b
  end
  b_grid = linspace(1e-7, max_b, grid_size)
  a_grid = linspace(1e-7, max_a, grid_size)
  return ToyModel(R, beta, alpha, Pi, inc, b_grid, a_grid, ut, gamma)
end

#==========================#
# Re-usable functions
#==========================#

# returns the derivative of the utility function
function get_du(mo::ToyModel)
  if mo.ut == "Log"
    du = (x) -> 1 / x
  elseif mo.ut == "CRRA"
    du = (x) -> x^(-mo.gamma)
  elseif mo.ut == "Exp"
    du = (x) -> Exp(-mo.gamma * x)
  elseif mo.ut == "Quad"
    du = (x) -> 1 - mo.gamma * x
  end
  return du
end

# returns the inverse of the derivative.
function get_idu(mo::ToyModel)
  if mo.ut == "Log"
    idu = (x) -> 1 / x
  elseif mo.ut == "CRRA"
    idu = (x) -> (1 / x)^(1 / mo.gamma)
  elseif mo.ut == "Exp"
    idu = (x) -> - log(x) / mo.gamma
  elseif mo.ut == "Quad"
    idu = (x) -> (1 - x) / mo.gamma
  end
  return idu
end

# for a sorted vector, find the indeces of its first elements that brackets x
function findindex_bound(vect::Vector, x::Real)
  if x > maximum(vect)
    # NOTE: if x exceeds the max. of the vector, then returns the two highest indeces
    x0, x1 = Void, length(vect)
  else
    for i in 1:(length(vect)-1)
      if (vect[i] <= x) && (vect[i+1] > x)
        x0 = i
        x1 = i + 1
        break
      end
    end
  end
  return x0, x1
end

# linearly interpolate a function F, defined on X, evaluated at xi
function linear_itp(xi::Float64, X::Vector, F::Vector)
  x0, x1 = findindex_bound(X, xi)
  if x0 == Void
    x0     = x1 - 1
    f1i    = F[x1] + (xi - X[x1]) * (F[x1] - F[x0]) / (X[x1] - X[x0])
  else
    f1i    = F[x0] + (xi - X[x0]) * (F[x1] - F[x0]) / (X[x1] - X[x0])
  end
  return f1i
end

# interpolate the young's policy function in the transfer dimension, with a
# lookup table for the income, and return the derivative
function der_pol(mo::ToyModel, x::AbstractMatrix)
  siz      = size(x)
  itp      = interpolate(x, (BSpline(Linear()), NoInterp()), OnGrid())
  itp      = scale(itp, mo.b_grid, 1:siz[2])
  d        = (xi, yi) -> gradient(itp, xi, yi)[1]
  return d
end

#==========================#
# EGM functions
#==========================#

function comp_y!(mo::ToyModel, _cy::Matrix, _co::Matrix)

  du, idu    = get_du(mo), get_idu(mo)
  C, bs      = zeros(size(_cy)), zeros(size(_cy))

  # Compute the right hand side of the FOC and invert it to obtain the consumption
  function foc_solver(a_idx::Integer, y_idx::Integer)
    T    = 0
    for k in 1:length(mo.inc)
      T += mo.Pi[y_idx, k] * du(_co[a_idx, k])
    end
    T   *= mo.beta * mo.R
    ce   = idu(T)
    b    = ce + mo.a_grid[a_idx] - mo.inc[y_idx]
    return ce, b
  end

  # Convert the function from the endo grid onto the initial grid
  function C_itp(b::Float64, y::Float64, yi::Integer)
    if b < bs[1, yi]
      cy  = b + y - mo.a_grid[1]
    else
      cy  = linear_itp(b, bs[:,yi], C[:,yi])
    end
    return cy
  end

  for i in 1:length(mo.a_grid)
    for j in 1:length(mo.inc)
      C[i,j], bs[i,j] = foc_solver(i, j)
    end
  end

  for (k, b_k) in enumerate(mo.b_grid)
    for (j, y_j) in enumerate(mo.inc)
      _cy[k,j] = C_itp(b_k, y_j, j)
    end
  end

end

function comp_o(mo::ToyModel, _cy::Matrix, _co::Matrix)

  du, idu     = get_du(mo), get_idu(mo)
  C, as, co   = zeros(size(_co)), zeros(size(_co)), zeros(size(_co))
  dcy         = der_pol(mo, _cy)

  # compute the right hand side of the FOC and invert it
  function foc_solver(k::Integer, j::Integer)
    O   = mo.alpha * dcy(mo.b_grid[k], j) * du(_cy[k, j])
    ce  = idu(O)
    a   = (ce + mo.b_grid[k]) / mo.R
    return ce, a
  end

  # convert the function from the endo grid onto the initial grid
  function C_itp(a::Float64, yi::Integer)
    if a < as[1,yi]
      # corner solution
      c = mo.R * a - mo.b_grid[1]
    else
      # interior solution
      c = linear_itp(a, as[:,yi], C[:,yi])
    end
    return c
  end

  for k in 1:length(mo.b_grid)
    for j in 1:length(mo.inc)
      # solve for C and as
      C[k,j], as[k,j] = foc_solver(k, j)
    end
  end

  for (i, a_i) in enumerate(mo.a_grid)
    for (j, y_j) in enumerate(mo.inc)
      co[i,j]   = C_itp(a_i, j)
    end
  end

  return co

end

function init(mo::ToyModel, isRand::Bool)
  if isRand
    co      = randn(length(mo.a_grid), length(mo.inc))
    cy      = randn(length(mo.b_grid), length(mo.inc))
  else
    co      = zeros(length(mo.a_grid), length(mo.inc))
    cy      = zeros(length(mo.b_grid), length(mo.inc))
    for (y_idx, y) in enumerate(mo.inc)
      co[:,y_idx] = mo.a_grid / 2
      cy[:,y_idx] = mo.beta * (mo.b_grid + y) / (1 + mo.beta)
    end
  end
  return cy, co
end

function ite(maxiter::Integer,
             mo::ToyModel;
             tol::Float64=1e-7,
             isCheck::Bool=false,
             isPlot::Bool=false)

  cy, co_0  = init(mo,isPlot)

  if isPlot
    CYplot      = zeros(length(mo.b_grid), 5)
    COplot      = zeros(length(mo.a_grid), 5)
    labels      = Array{String}(5)
    CYplot[:,1] = cy[:,1]
    COplot[:,1] = co_0[:,1]
    labels[1]   = "Init."
  end

  for iter in 1:maxiter
    comp_y!(mo, cy, co_0)
    co_1    = comp_o(mo, cy, co_0)
    err     = maxabs(co_0 - co_1)

    if isPlot && (iter <= 3)
      CYplot[:,iter+1] = cy[:,1]
      COplot[:,iter+1] = co_1[:,1]
      labels[iter+1]  = ""
    end

    if err < tol
      # println("EGM converged after $iter iterations")

      if isPlot
        CYplot[:,5] = cy[:,1]
        COplot[:,5] = co_1[:,1]
        labels[5]   = "P.$iter"
        C(g::ColorGradient) = RGB[g[z] for z=linspace(0,1,5)]
        g             = :matter
        colors        = C(cgrad(g))

        plot_1 = Plots.plot(mo.b_grid, CYplot, grid = false,
                            label = labels', color = colors', title = L"\mathcal{C}_y",
                            xlab = L"b", ylab = L"\mathcal{C}_y(b, \, y)")
        plot_2 = Plots.plot(mo.a_grid, COplot, grid = false,
                            label = labels', color = colors', title = L"\mathcal{C}_o",
                            xlab = L"a", ylab = L"\mathcal{C}_o(a, \, y)")
        display(Plots.plot(plot_1, plot_2))
      end

      # compute the two other Markov strategies
      a, b   = zeros(size(cy)), zeros(size(co_1))
      for (j, y) in enumerate(mo.inc)
        a[:,j] = y + collect(mo.b_grid) - cy[:,j]
        b[:,j] = mo.R * collect(mo.a_grid) - co_1[:,j]
      end

      return a, b, cy, co_1
      break
    elseif iter == maxiter
      error("EGM did not converge after $iter iterations")
    end

    if isCheck
      println("Iteration #$iter, error = ", err)
    end
    # update current guess
    co_0     = copy(co_1)
  end
end

#==========================#
# Solution EGM
#==========================#

"""
```
getSol(;R=1.01, beta=0.98, alpha=0.8, Pi=[0. 1.;1. 0.], inc = [1., 6.], grid_size = 100, ut = "Log", gamma = 1.)
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
                  inc = [1., 6.],
                  grid_size = 100,
                  ut = "Log",
                  gamma = 1.0)


  olg = ToyModel(R = R, beta = beta, alpha = alpha, Pi = Pi, inc = inc,
                grid_size = grid_size, ut = ut, gamma = gamma)
  sol = ite(30,olg)
  return sol
end

end
