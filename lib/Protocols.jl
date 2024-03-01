include("Aux.jl")

#this file contains data structures and functions used by the main code 


"""
`smith(system::SystemDescription, x::Vector{<:Real},p::Vector{<:Real}; λ=1.0, τ=1.0 ) -> Vector{<:Real}`

Smith dynamics. Takes parameters `system` (a SystemDescription), `x` (a vector with the current state of the population
and `p` the currently offered payoff vector.

# Examples
```julia
(TODO)
julia> g  = ...;
julia> cs = ...;
julia> sd = SystemDescription(g,cs);
julia> smith(sd, ones(ns),zeros(ns); λ=7.0, τ=3.0 )
...
```
""" 
function smith(system::SystemDescription, x::Vector{<:Real},p::Vector{<:Real}; λ=1.0, τ=1.0 )
    
    NS = length(x)
    sm(dp) = max(min(λ*dp, τ),0.0)
    normalize!(x,1)

    T = [ sm(p[j]-p[i]) for i=1:NS, j=1:NS ] 

    x_dot = (T'*diagm(x)-diagm(x)*T)*ones(NS) 

    x_dot
end



"""
`bnn(system::SystemDescription, x::Vector{<:Real},p::Vector{<:Real}; λ=1.0, τ=1.0 ) -> Vector{<:Real}`

Brown-von Neumann-Nash (BNN) dynamics. Takes parameters `system` (a SystemDescription), `x` (a vector with the current state of the population
and `p` the currently offered payoff vector.

# Examples
```julia
(TODO)
julia> g  = ...;
julia> cs = ...;
julia> sd = SystemDescription(g,cs);
julia> bnn(sd, ones(ns),zeros(ns); λ=7.0, τ=3.0 )
...
```
"""
function bnn(system::SystemDescription, x::Vector{<:Real},p::Vector{<:Real}; λ=1.0, τ=1.0 )
    NS = length(x)
    normalize!(x,1)

    p_hat = p .- x'*p
    
    T = min.(λ.*max.(p_hat,0),τ)  
    
    x_dot  = T*sum(x)-x*sum(T)

    x_dot
end


