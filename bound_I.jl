


#%%

#using Revise
using DifferentialEquations, LinearAlgebra, Plots
using ForwardDiff, JuMP, Ipopt, LaTeXStrings, Symbolics

include("lib/Aux.jl")

#the other k values go inside the sd::SystemDescription
function get_I_bound_given_k(SD::SystemDescription,i,αs,Z0)
    
    ret = []
    
    sd = copy(SD)
    for α in αs
        m = Model(Ipopt.Optimizer)
        set_attribute(m, "print_level", 0)
        local ns = sd.game_param.ns
        sd.game_param.k[i] = α
        y0,x0,p0 = Z0

        #without an initial guess the optimizer threw an error
        @variable(m, I>=0.00001, start = .001)
        @variable(m, R>=0.0, start = .001)
        @variable(m, x[1:ns]>=0, start = 1/ns)

        @constraint(m, I+R<=1)
        @constraint(m, sum(x)==1)
        @constraint(m, L_noS(sd, I, R, x, zeros(ns))
                        <=L_noS(sd, y0[1], y0[2], x0, p0) )

        @objective(m, Max, I)


        optimize!(m)

        push!(ret, (value.(I),[value.(I),value.(R),value.(x)]) )
    end

    ret
end




