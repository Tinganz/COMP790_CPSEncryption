module Experiment
export matrix_integral, evolve, generate_uncertainty, ideal_evolve, online_evolve

using QuadGK
using Distributions
using ControlSystemsBase
using LinearAlgebra
using ReachabilityAnalysis

function matrix_integral(A::AbstractMatrix, B::AbstractMatrix, lower_limit::Float64, upper_limit::Float64)
  integrand = s -> exp(A * s) * B
  result, _ = quadgk(integrand, lower_limit, upper_limit)
  return result
end

function generate_uncertainty(σ1::Float64, σ2::Float64, μ::Float64)
  dist_λ1 = Normal(μ, σ1)
  dist_λ2 = Normal(μ, σ2)
  λ11 = clamp(rand(dist_λ1), -1, 1)
  λ12 = clamp(rand(dist_λ1), -1, 1)
  λ21 = clamp(rand(dist_λ2), -1, 1)
  λ22 = clamp(rand(dist_λ2), -1, 1)
  return [λ11 λ12 λ21 λ22]
end

function evolve(A::AbstractMatrix, B::AbstractMatrix, K::AbstractMatrix, H::Integer, z0::Vector{Float64}, u1_0::Float64, u2_0::Float64, σ1::Float64, σ2::Float64, μ::Float64, n::Integer)
  u0 = [u1_0; u2_0]
  z = Vector{typeof(z0)}(undef, H + 1)
  u = Vector{typeof(u0)}(undef, H)
  z[1] = z0
  u[1] = u0
  z[2] = A * z[1] + B * u[1]
  z[2][end-length(u2_0)+1:end, :] .= 0
  λ21 = nothing
  λ22 = nothing
  z_fix = z[2]
  for k in 2:H
    λ = generate_uncertainty(σ1, σ2, μ)
    λ11 = λ[1]
    λ12 = λ[2]
    if mod((k-2), n) == 0
      λ21 = λ[3]
      λ22 = λ[4]
      z_fix = z[k]
    end
    K_error = [K[1, :][1]*(1+λ11) K[1, :][2]*(1+λ12) K[1, :][3]; K[2, :][1]*(1+λ21) K[2, :][2]*(1+λ22) K[2, :][3]]
    u[k] = -[dot(K_error[1, :], z[k]);
         dot(K_error[2, :], z_fix)]
    z[k+1] = A * z[k] + B * u[k]
  end
  return z, u
end

function online_evolve(A::AbstractMatrix, B::AbstractMatrix, K::AbstractMatrix, H::Integer, z0::Vector{Float64}, u1_0::Float64, u2_0::Float64, σ1::Float64, σ2::Float64, μ::Float64, ideal::Vector{Vector{Float64}}, threshold::Float64)
  u0 = [u1_0; u2_0]
  z = Vector{typeof(z0)}(undef, H + 1)
  u = Vector{typeof(u0)}(undef, H)
  z[1] = z0
  u[1] = u0
  z[2] = A * z[1] + B * u[1]
  z[2][end-length(u2_0)+1:end, :] .= 0
  λ21 = nothing
  λ22 = nothing
  z_fix = z[2]
  for k in 2:H
    λ = generate_uncertainty(σ1, σ2, μ)
    λ11 = λ[1]
    λ12 = λ[2]
    if (k == 2) 
      λ21 = λ[3]
      λ22 = λ[4]
    end
    if norm(ideal[k] - z[k][1:2]) > threshold
      λ21 = λ[3]
      λ22 = λ[4]
      z_fix = z[k]
    else
    end
    K_error = [K[1, :][1]*(1+λ11) K[1, :][2]*(1+λ12) K[1, :][3]; K[2, :][1]*(1+λ21) K[2, :][2]*(1+λ22) K[2, :][3]]
    u[k] = -[dot(K_error[1, :], z[k]);
         dot(K_error[2, :], z_fix)]
    z[k+1] = A * z[k] + B * u[k]
  end
  return z, u
end

function ideal_evolve(A::AbstractMatrix, B::AbstractMatrix, K::AbstractMatrix, H::Integer, x0::Vector{Float64}, u0::Float64)
  u0 = [u0]
  x = Vector{typeof(x0)}(undef, H + 1)
  u = Vector{typeof(u0)}(undef, H + 1)
  x[1] = x0
  u[1] = u0
  for k in 1:H
    x[k+1] = A * x[k] + B * u[k]
    u[k+1] = - K * x[k+1]
  end
  return x, u
end
end
