include("Aux.jl")
include("Protocols.jl")


#this file contains data structures and functions used by the main code 





"""
`edm!(du,u,param,t;learning_rule=smith) -> Array{Float}`

total dynamics of our model, uses smith(p,u,t,i;λ=0.1,τ=0.1) as the protocol by default
""" 
function edm!(du,u,param,t;learning_rule=smith)

    Z = toState(param, u)
    y,x,q = Z

    y = abs.(y) #for the SIRS this is ok

    normalize!(x,1)
    x .= abs.(x) #for any pop game this is ok

    u .= toVec(param,[y,x,q])
    Z = toState(param, u)

    dZ = zero.(Z)

    dZ[1] .= param.cs_param.f(param, Z, 0.0)
    dZ[2] .= learning_rule(param,x,param.game_param.F(param, Z, 0.0))
    dZ[3] .= param.game_param.G(param,Z,0.0) 

    du .= toVec(param, dZ)

end



