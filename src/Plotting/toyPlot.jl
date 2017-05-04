module ToyPlot

include("./../ToyModel/EGM.jl")
include("./../ToyModel/PFI.jl")
include("./../ToyModel/VFI.jl")
using Plots, LaTeXStrings

function cf(;save::Bool=false, isPFI::Bool=true)

  if isPFI
    mo             = ToyPFI.ToyModel(grid_size = 20,max_a = 6.)
    sol            = ToyPFI.ite(20,mo)
    y_p, y_r       = mo.inc_vals
    a_grid, b_grid = mo.asset_grid, mo.transfer_grid
  else
    mo             = ToyEGM.ToyModel(grid_size = 20,max_a = 6.)
    sol            = ToyEGM.ite(20,mo)
    y_p, y_r       = mo.inc
    a_grid, b_grid = mo.a_grid, mo.b_grid
  end
  beta, al, R      = mo.beta, mo.alpha, mo.R

  # build the Markov strategy
  a_p(b::Float64) = beta * (y_p + b) / (1 + beta)
  b_p(a::Float64) = 0.
  function b_r(a::Float64)
      if al * R * a > y_p
          return (al * R * a - y_p) / (1 + al)
      else
          return 0
      end
  end
  function a_r(b::Float64)
      w     = b + y_r
      if w <= (1 + beta) * y_p / (al * beta * R)
          return beta * w / (1 + beta)
      else
          return (beta * R * w * (1 + al) - y_p) / (R * (1 + beta * (1 + al)))
      end
  end

  # Equilibrium level
  as_r = [(y_r * beta * R * (1 + al) - y_p) / (R * (1 + beta * (1 + al)))]
  as_p = [al * beta * beta * (y_p + R * y_r) / ((1 + beta) * 1 + beta * (1 + al))]
  bs_r = [0.]
  bs_p = [(al * beta * R * y_r - (1 + beta) * y_p) / (1 + beta * (1 + al))]

  C(g::ColorGradient) = RGB[g[z] for z=linspace(0,1,5)]
  g      = :reds
  colors = C(cgrad(g))

  plot_a_r = Plots.plot(b_grid, sol[1][:,2], label = "Num.", line = (2, :solid),
                        legendfont = font(6), title = "Savings rich", titlefont = font(10),
                        xlab = L"b", grid = false, color = colors[2])
  Plots.plot!(b_grid, a_r, line = (2, :dash), label = "Analytical", color = colors[3])
  Plots.scatter!(bs_r, a_r, markersize = 5, label = "Eq.", color = colors[4])

  plot_a_p = Plots.plot(b_grid, sol[1][:,1], line = (2, :solid), label = "Num.", legendfont = font(6),
            title = "Savings poor", titlefont = font(10), xlab = L"b",
            grid = false, color = colors[2])
  Plots.plot!(b_grid, a_p, line = (2, :dash), label = "Analytical", color = colors[3])
  Plots.scatter!(bs_p, a_p, markersize = 5, label = "Eq.", color = colors[4])

  plot_b_p = Plots.plot(a_grid, sol[2][:,2], line = (2, :solid), label = "Num.", legendfont = font(6),
            ylim = [-0.3;0.3], title = "Transfer to rich", titlefont = font(10), xlab = L"a",
            grid = false, color = colors[2])
  Plots.plot!(a_grid, b_p, line = (2, :dash), label = "Analytical", color = colors[3])
  Plots.scatter!(as_p, b_p, markersize = 5, label = "Eq.", color = colors[4])

  plot_b_r = Plots.plot(a_grid, sol[2][:,1], line = (2, :solid), label = "Num.", legendfont = font(6),
            title = "Transfer to poor", titlefont = font(10), xlab = L"a",
            grid = false, color = colors[2])
  Plots.plot!(a_grid, b_r, line = (2, :dash), label = "Analytical", color = colors[3])
  Plots.scatter!(as_r, b_r, markersize = 5, label = "Eq.", color = colors[4])

  plot_t   = Plots.plot(plot_a_p, plot_a_r, plot_b_p, plot_b_r, layout = 4)
  if save
    Plots.savefig("num_vs_cf.pdf")
  end

  return plot_t
end

function accuracyVFI()
  mod_v = ToyVFI.ToyModel(grid_size=100)
  sol_v = ToyVFI.ite(100,mod_v)
  mod_p = ToyPFI.ToyModel(grid_size=100)
  sol_p = ToyPFI.ite(30,mod_p)
  C(g::ColorGradient) = RGB[g[z] for z=linspace(0,1,4)]
  g     = :reds
  colors= C(cgrad(g))

  ## Compare the two solutions
  # Assets
  plot_1 = Plots.plot(mod_v.b_grid, sol_v[1], label = ["VFI, poor" "VFI, rich"], title = "Savings",
                      xlab = L"b", grid = false, line = :dash, color = [colors[1] colors[3]])
  Plots.plot!(mod_p.transfer_grid, sol_p[1], label = ["Ana., poor" "Ana., rich"], color = [colors[2] colors[4]])
  # Transfer
  plot_2 = Plots.plot(mod_v.a_grid, sol_v[2], label = ["VFI, rich" "VFI, poor"], title = "Transfer",
                      xlab = L"a", grid = false, line = :dash, color = [colors[3] colors[1]])
  Plots.plot!(mod_p.asset_grid, sol_p[2], label = ["Ana., rich" "Ana., poor"], color = [colors[4] colors[2]])
  Plots.plot(plot_1, plot_2)
end

function effiency(;grid_max=500)
  max_g = [20:100:grid_max;500]
  vfi_t = zeros(length(max_g))
  pfi_t = zeros(length(max_g))
  egm_t = zeros(length(max_g))

  # pre-compile the algorithm
  ToyPFI.ite(30,ToyPFI.ToyModel(grid_size=20))
  ToyEGM.ite(30,ToyEGM.ToyModel(grid_size=20))
  ToyVFI.ite(100,ToyVFI.ToyModel(grid_size=20))

  for (i, a) in enumerate(max_g)
    println("Currently @ $i out of $(length(max_g))")
    pfi_t[i] = @elapsed ToyPFI.getSol(grid_size=a);
    egm_t[i] = @elapsed ToyEGM.getSol(grid_size=a);

    t = @elapsed sol = ToyVFI.getSol(grid_size=a);
    if typeof(sol) <: Tuple
      vfi_t[i] = t
    else
      vfi_t[i] = NaN 
    end
  end

  return Plots.scatter(max_g, [vfi_t, pfi_t, egm_t], label = ["VFI" "PFI" "EGM"],
                      xlab = "Grid size", ylab = "Time (s)")
end

end
