module Experiment
export matrix_integral, evolve, generate_uncertainty, ideal_evolve, flip_bits

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

function flip_bits(a::Float64, b::Int)
  bits = bitstring(a)
  n = lastindex(bits)
  flipped = String([i > n - b ? (bits[i] == '0' ? '1' : '0') : bits[i] for i in 1:n])
  flipped_int = parse(UInt64, flipped; base=2)
  new_a = reinterpret(Float64, flipped_int)
  return new_a
end

function flip_bits_vec(v::AbstractVector{Float64}, b::Int)
  return [flip_bits(a, b) for a in v]
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

function evolve(A::AbstractMatrix, B::AbstractMatrix, K::AbstractMatrix, H::Integer, z0::Vector{Float64}, u1_0::Float64, u2_0::Float64, unprotected::Integer)
  u0 = [u1_0; u2_0]
  z = Vector{typeof(z0)}(undef, H + 1)
  u = Vector{typeof(u0)}(undef, H)
  z[1] = z0
  u[1] = u0
  z[2] = A * flip_bits_vec(z[1], unprotected) + B * flip_bits_vec(u[1], unprotected)
  z[2][end-length(u2_0)+1:end, :] .= 0
  for k in 2:H
    u[k] = - dot(K, flip_bits_vec(z[k], unprotected));
    z[k+1] = A * flip_bits_vec(z[k], unprotected) + B * flip_bits_vec(u[k], unprotected)
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
