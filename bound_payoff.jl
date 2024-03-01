


#%%

#using Revise
using DifferentialEquations, LinearAlgebra, Plots
using ForwardDiff, JuMP, Ipopt, LaTeXStrings, Symbolics

include("lib/Aux.jl")



#find the grad_x of U(y;x)
function get_objective(sd::SystemDescription)
    local ns = sd.game_param.ns

    Symbolics.@variables X[1:ns],II,RR  

    symU = Symbolics.gradient(L_noS(sd, II, RR, [X...],zero(ns)),[X...])/sd.game_param.k[3]+sd.game_param.c
    symU = Symbolics.simplify.(symU,threaded=true, simplify_fractions=false)

    DsymU = Base.remove_linenums!(build_function(symU, II, RR, X, expression=Val{false})[1])

    #run the function as a sanity check
    DsymU(1/3,1/3,[1.0;zeros(ns-1)])
    DsymU
end


#the k values go inside the sd::SystemDescription
function get_p_bound_given_k(sd::SystemDescription,Z0)
    DsymU = get_objective(sd)

    local ns = sd.game_param.ns

    solutions = Float64[]
    my_lock = Threads.ReentrantLock();

    for i = 1:ns
        m = Model(Ipopt.Optimizer)
        set_attribute(m, "print_level", 0)
        local y0,x0,p0 = Z0

        #without an initial guess the optimizer threw an error
        @variable(m, I>=0.0001, start = .001)
        @variable(m, R>=0, start = .001)
        @variable(m, x[1:ns]>=0, start = 1/ns)

        @constraint(m, I+R<=1)
        @constraint(m, sum(x)==1)
        @constraint(m, L_noS(sd, I, R, x,zeros(1:ns))<=
                        L_noS(sd, y0[1], y0[2], x0, p0))

    

        @objective(m, Max, abs(DsymU(I,R,x)[i]) )


        optimize!(m)
        Threads.lock(my_lock) do
            push!(solutions, objective_value(m))
        end
    end
    maximum(solutions)
end



