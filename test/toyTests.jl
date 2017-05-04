module toyTests

using Base.Test
include("./../src/ToyModel/PFI.jl")
include("./../src/ToyModel/EGM.jl")
include("./../src/ToyModel/VFI.jl")

# What we need to tests
# 1. There is indeed some update
# 2. The results correspond to the analical solution
# 3. The algorithm converges

@testset "re-usable functions" begin
  @testset "u, u' and inverse u'" begin
    # same functions for PFI, VFI and EGM
    mod = ToyEGM.ToyModel(grid_size=20)
    du  = ToyEGM.get_du(mod)
    idu = ToyEGM.get_idu(mod)
    @test du(3.)  == 1/3
    @test idu(3.) == 1/3

    mod = ToyEGM.ToyModel(grid_size=20, ut = "CRRA", gamma = 2.)
    du  = ToyEGM.get_du(mod)
    idu = ToyEGM.get_idu(mod)
    @test du(3.)  == 3.^(-mod.gamma)
    @test idu(3.) == (1/3)^(1/mod.gamma)
  end

  @testset "interpolant" begin
    # only for PFI and VFI
    mod    = ToyPFI.ToyModel(grid_size=20)
    A1     = zeros(length(mod.transfer_grid),length(mod.inc_vals))
    B1     = zeros(length(mod.asset_grid),length(mod.inc_vals))
    for i in 1:length(mod.inc_vals)
      A1[:,i]  = mod.R * (mod.transfer_grid + mod.inc_vals[i]) / (1 + mod.R)
      B1[:,i]  = mod.alpha^2 * mod.asset_grid
    end

    A1_itp, B1_itp = ToyPFI.itp_pol(mod,'a',A1), ToyPFI.itp_pol(mod,'b',B1)
    @test A1[1,1] - 1e-9 <= A1_itp[mod.transfer_grid[1],1] <= A1[1,1] + 1e-9
    for i in 1:5:20
      @test A1[i,1] - 1e-9 <= A1_itp[mod.transfer_grid[i],1] <= A1[i,1] + 1e-9
      @test A1[i,2] - 1e-9 <= A1_itp[mod.transfer_grid[i],2] <= A1[i,2] + 1e-9
      @test B1[i,1] - 1e-9 <= B1_itp[mod.asset_grid[i],1] <= B1[i,1] + 1e-9
      @test B1[i,2] - 1e-9 <= B1_itp[mod.asset_grid[i],2] <= B1[i,2] + 1e-9
    end
  end

  @testset "EGM specific functions" begin
    X = collect(1:1:20)
    Y = [1;1;3;5;7]
    @test (3,4)            == ToyEGM.findindex_bound(X,3)
    @test (Void,length(X)) == ToyEGM.findindex_bound(X,22)
    @test (2,3)            == ToyEGM.findindex_bound(Y,1)

    # linear function => linear interpolation should get it right
    f(x) = 2x + 1
    xx   = linspace(0.,20.,8)
    F    = f.(xx)
    # test both inside and outside the grid
    @test f(2.) == ToyEGM.linear_itp(2.,collect(xx),F)
    @test f(30.) == ToyEGM.linear_itp(30.,collect(xx),F)
  end
end

@testset "initialization" begin
  # PFI
  mod    = ToyPFI.ToyModel(grid_size=20)
  A, B = ToyPFI.init_values(mod,false)
  @test size(A) == (length(mod.transfer_grid),length(mod.inc_vals))
  @test size(B) == (length(mod.asset_grid),length(mod.inc_vals))
  # VFI
  mod        = ToyVFI.ToyModel(grid_size=20)
  V0, A, B = ToyVFI.init_values(mod)
  @test size(A) == (length(mod.b_grid),length(mod.inc_vals))
  @test size(B) == (length(mod.a_grid),length(mod.inc_vals))
  # EGM
  mod    = ToyEGM.ToyModel(grid_size=20)
  A, B = ToyEGM.init(mod,false)
  @test size(A) == (length(mod.b_grid),length(mod.inc))
  @test size(B) == (length(mod.a_grid),length(mod.inc))
end

@testset "update" begin
  @testset "PFI" begin
    mod    = ToyPFI.ToyModel(grid_size=20)
    A1, B1 = ToyPFI.init_values(mod,false)
    A2     = copy(A1)
    ToyPFI.young!(B1,A2,mod)
    B2 = ToyPFI.old(B1,A2,mod)
    # check that indeed, the algorithm updated the solutions
    # note that we cannot do pointwise checking, as some of the cells might not have moved
    @test mean(A1[:,1]) != mean(A2[:,1])
    @test mean(A1[:,2]) != mean(A2[:,2])
    @test mean(B1[:,1]) != mean(B2[:,1])
    @test mean(B1[:,2]) != mean(B2[:,2])
  end

  @testset "VFI" begin
    mod          = ToyVFI.ToyModel(grid_size=20)
    V0_1, A1, B1 = ToyVFI.init_values(mod)
    A2, B2       = copy(A1), copy(B1)
    ToyVFI.young!(V0_1,A2,mod)
    V0_2 = ToyVFI.old!(V0_1,A2,B2,mod)
    @test mean(A1[:,1])   != mean(A2[:,1])
    @test mean(A1[:,2])   != mean(A2[:,2])
    @test mean(B1[:,1])   != mean(B2[:,1])
    @test mean(B1[:,2])   != mean(B2[:,2])
    @test mean(V0_1[:,1]) != mean(V0_2[:,1])
    @test mean(V0_1[:,2]) != mean(V0_2[:,2])
  end

  @testset "EGM" begin
    mod = ToyEGM.ToyModel(grid_size=20)
    CY0, CO0 = ToyEGM.init(mod,false)
    CY1 = copy(CY0)
    ToyEGM.comp_y!(mod,CY1,CO0)
    CO1 = ToyEGM.comp_o(mod,CY1,CO0)
    @test mean(CY0[:,1]) != mean(CY1[:,1])
    @test mean(CY0[:,2]) != mean(CY1[:,2])
    @test mean(CO0[:,1]) != mean(CO1[:,1])
    @test mean(CO0[:,2]) != mean(CO1[:,2])
  end
end

@testset "results" begin
  @testset "VFI" begin
    # test of convergence
    mod = ToyVFI.ToyModel(grid_size=20)
    sol = ToyVFI.ite(100,mod)
    # if did not converge, then return Void
    @test typeof(sol) <: Tuple{Array,Array}
  end

  mo               = ToyPFI.ToyModel(grid_size=20)
  beta, al, R      = mo.beta, mo.alpha, mo.R
  y_p, y_r       = mo.inc_vals
  a_grid, b_grid = mo.asset_grid, mo.transfer_grid

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

  # point at which to evaluate the solution
  a = [1;5;10;15;20]
  b = [1;5;10;15;20]

  @testset "PFI" begin
    # test of convergence
    mod = ToyPFI.ToyModel(grid_size=20)
    sol = ToyPFI.ite(20,mod)
    # if did not converge, then return Void
    @test typeof(sol) <: Tuple{Array,Array,Array,Array}
    # then test if solutions are equal to the analytical ones
    for i in 1:length(a)
      @test a_p(mod.transfer_grid[i]) - 1e-9 <= sol[1][i,1] <= a_p(mod.transfer_grid[i]) + 1e-9
      @test a_r(mod.transfer_grid[i]) - 1e-9 <= sol[1][i,2] <= a_r(mod.transfer_grid[i]) + 1e-9
      @test b_r(mod.asset_grid[i]) - 1e-9 <= sol[2][i,1] <= b_r(mod.asset_grid[i]) + 1e-9
      @test b_p(mod.asset_grid[i]) - 1e-9 <= sol[2][i,2] <= b_p(mod.asset_grid[i]) + 1e-9
    end
  end

  @testset "EGM" begin
    # convergence test
    mod = ToyEGM.ToyModel(grid_size=20)
    sol = ToyEGM.ite(20,mod)
    @test typeof(sol) <: Tuple{Array,Array,Array,Array}
    # NOTE: apparenlty, bug in the code. Solutions do not match no more
    # TODO: check if time
  end
end

end
