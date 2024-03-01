

"""
`mutable struct Game`

F: calculates the current payoff, a function (sd::SystemDescription,state::Vector{Vector{Real}},t::Real) -> Vector{Real}
G: payoff dynamics, a function (sd::SystemDescription,state::Vector{Vector{Real}},t::Real) -> Vector{Real}
""" 
mutable struct Game
    ns::Int              #number of strategies
    nq::Int              #number of states of the dynamic payoff 
    F                    #payoff function (sd::SystemDescription,state::Vector{Vector{Real}},t::Real)
    c::Vector{Real}      #cost vector
    c_star::Real         #available budget
    x_star::Vector{Real} #target equilibrium
    p_star::Vector{Real} #target payoff
    k                    #pdm param
    G                    #pdm dynamics
end

"""
`mutable struct CoupledSystem`

f: dynamics of the coupled system, a function (sd::SystemDescription,Z,t) -> Vector{Real}
""" 
mutable struct CoupledSystem
    n::Int              #number of strategies 
    f                   #dynamics
    parameters::Tuple   #tuple with the parameters of the cs
end


mutable struct SystemDescription
    game_param::Game          #parameters of the game
    cs_param::CoupledSystem   #parameters of the coupled dynamical system
end


import Base.copy

"""
`copy(g::Game)`

deep copy of variable `g`.
""" 
function copy(g::Game)
    Game(g.ns,g.nq,g.F,g.c,g.c_star,copy(g.x_star),
        copy(g.p_star),copy(g.k),g.G )
end

"""
`copy(cs::CoupledSystem)`

deep copy of variable `cs`.
""" 
function copy(cs::CoupledSystem)
    CoupledSystem(cs.n,cs.f,cs.parameters)
end

"""
`copy(sd::SystemDescription)`

deep copy of variable `sd`.
""" 
function copy(sd::SystemDescription)
    SystemDescription(copy(sd.game_param),copy(sd.cs_param))
end

"""
`simplex_plot!(s;ptype=scatter!)`

Generate a plot of the strategies evolving on the simplex, games must have 3 strategies.
""" 
function simplex_plot!(s;ptype=scatter!)

    if s isa Vector
        for sol in s
            simplex_plot_aux(sol; ptype=ptype)
        end
    else
        simplex_plot_aux(s; ptype=ptype)
    end
    plot!([0; 1; -1; 0],[1; -1/2 ;-1/2; 1], ticks=false, label=false)
end

"""
see simplex_plot!
""" 
function simplex_plot_aux(s; ptype=scatter!)
    t = s.t
    x = s.u

    x = map(v->[0 1 -1;1 -1/2 -1/2]*v,x)
    x = hcat(x...)
    ptype(x[1,:],x[2,:],label=false,lw=3)
    scatter!([x[1,end]],[x[2,end]],ms=10,shape=:star,label=false,showaxis = false)
end


"""
`toVec(sd, Z::Vector{Vector{T}}) where {T<:Real} -> Vector{<:Real}`

The function to convert a state object `Z` into a Vector{<:Real} based on the description in `sd`.
""" 
function toVec(sd, Z::Vector{Vector{T}}) where {T<:Real}
    vcat(Z...)
end

"""
`toState(sd, Z::Vector{Vector{T}}) where {T<:Real} -> Vector{Vector{T}}`

The function to convert a vector `Z` into a state object based on the description in `sd`.
""" 
function toState(sd, Z::Vector{T}) where {T<:Real}
    if sd isa SystemDescription
        ny = sd.cs_param.n
        nx = sd.game_param.ns
        nq = sd.game_param.nq
        [Z[1:ny],Z[ny.+(1:nx)],Z[ny.+nx.+(1:nq)]]
    else
        [Z[1:2],Z[3:5],Z[6:8]]
    end
end


#checking if the functions work 
begin
    z0 = collect(0:7)
    @assert all(toVec([],toState([],z0)).==z0)
    z0 = toState([],z0)
    @assert all(toState([],toVec([],z0)).==z0)
end


"""
`Ib(sd::SystemDescription,B) -> <:Real`

The function to calculate the endemic proportion of infected. Takes parameters `sd` (a SystemDescription) and 
    `B` (a scalar) with the current transmission rate.

# Examples
```julia
(TODO)
julia> g  = ...;
julia> cs = ...;
julia> sd = SystemDescription(g,cs);
julia> Ib(sd, 0.1 )
...
```
""" 
function Ib(sd::SystemDescription,B)
    local _,δ,σ,γ,ω = sd.cs_param.parameters

    bB = γ*B+ω*(B-δ)+δ*(B-σ)
    Δ  = bB^2-4*δ*ω*(B-δ)*(B-σ)

    #our method requires a positive endemic equilibrium, but numeric error might calculate an
    # equilibrium that is nonpositive if B(x) is close to \sigma. So we use this to avoid such errors 
    if bB-sqrt(Δ) isa AbstractFloat
        if bB-sqrt(Δ)<=1e-5
            1e-5/(2*δ*(B-δ))
        else
            (bB-sqrt(Δ))/(2*δ*(B-δ))
        end
    else
        (bB-sqrt(Δ))/(2*δ*(B-δ))
    end

end

"""
`Rb(sd::SystemDescription,B) -> <:Real`

The function to calculate the endemic proportion of recovered. Takes parameters `sd` (a SystemDescription) and 
    `B` (a scalar) with the current transmission rate.

# Examples
```julia
(TODO)
julia> g  = ...;
julia> cs = ...;
julia> sd = SystemDescription(g,cs);
julia> Rb(sd, 0.1 )
...
```
""" 
function Rb(sd::SystemDescription,B)
    local _,δ,σ,γ,ω = sd.cs_param.parameters
    (1-σ/B)-(1-δ/B)*Ib(sd,B)
end

#

"""
`L_noS(sd::SystemDescription, I, R, x, p) -> <:Real`

This is (almost) a Lyapunov function for the overall system. 
It does not include the delta-passive storage function of the EDM, which is dependent 
on the protocol being used by the population.

# Examples
```julia
(TODO)
julia> g  = ...;
julia> cs = ...;
julia> sd = SystemDescription(g,cs);
julia> L_noS(sd, 0.01, 0.1, [0.1;0.8;0.1], zeros(3))
...
```
""" 
function L_noS(sd::SystemDescription, I, R, x, p)
    
    x_star = sd.game_param.x_star
    p_star = sd.game_param.p_star
    local B = sd.cs_param.parameters[1]
    local k1,k2,k3 = sd.game_param.k

    a(x) = B(x)/(γ+δ*Rb(sd,B(x)))
    U(I,R,x) = (I-Ib(sd,B(x)))+Ib(sd,B(x))*log(Ib(sd,B(x))/I)+a(x)*(R-Rb(sd,B(x)))^2/2

    k1*U(I,R,x)+k2*(x-x_star)'*(x-x_star)/2+k3*(-x'*p_star.+max(p_star...)) 
end

