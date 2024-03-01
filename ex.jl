


#%%

#using Revise
using DifferentialEquations, LinearAlgebra, Plots
using ForwardDiff, JuMP, Ipopt, LaTeXStrings

#plotly()
gr()
   
USE_DEFED = @isdefined USE_TIKZ 
if USE_DEFED && (USE_TIKZ)
    pgfplotsx()
    plt_ext = "tikz"
else
    #plotly()
    gr()
    plt_ext = "png"
end

if @isdefined DEF_CONFIG
    plt_size = (800,600)
    default(;DEF_CONFIG...)
else
    plt_size = (600,500)
    #plt_size = (900,900)
    #default(linewidth = 3, markersize=10, margin = 10*Plots.mm,
    #    tickfontsize=12, guidefontsize=20, legend_font_pointsize=18)

end

include("common_init.jl")


function do_sim(sd::SystemDescription,z0)
    plot()

    T = 600.0
    f1 = plot()
    f2 = plot()
    f3 = plot()

    Pl = [0.5,1.0,2.0]
    Pt = [0.05,0.1,0.2]
    
    learning_rules=vcat(
        [((s,x,p)->smith(s,x,p;λ=l,τ=t), L"Smith \lambda="*"$l"*L"\tau="*"$t") for l in Pl for t in Pt],
        [((s,x,p)->bnn(s,x,p;λ=l,τ=t), L"BNN \lambda="*"$l"*L"\tau="*"$t") for l in Pl  for t in Pt]
    )
    #=
    learning_rules=[
        ((s,x,p)->smith(s,x,p;λ=10.0,τ=0.1), L"Smith \lambda=10.0"),
        ((s,x,p)->bnn(s,x,p;λ=0.1,τ=0.1), L"BNN \lambda=0.001"),
    ]=#

    for (lr,name) in learning_rules
        my_edm!(du,u,p,t) = edm!(du,u,p,t;learning_rule=lr)

        prob = ODEProblem(my_edm!,z0,[0.0,T],sd)
        sol = solve(prob, alg_hints=[:stiff])

        solState = [ B(toState(sd, sol(t))[2]) for t=range(extrema(sol.t)...,1000)]


        co = palette(:default)
        plot!(f1,sol,idxs=1:1,label=false, color=co[1] )
        plot!(f2,sol,idxs=3:5,label=false, color=co[3:5]' )

        Y = map(x->x[3:5]'*(x[6:8]+mos.game_param.c)  ,  sol)
        plot!(f3,sol.t,Y,label=false, color=co[7] )
        #plot!(f3,sol,idxs=6:8,label=false, color=co[3:5]' )

    end



    co = palette(:default)


    plot!(f1,[1],[0.06],label=false,  #L"I",
            color=co[1], ylabel=L"I(t)",
            legendfont=16, leg=:outerright)
    plot!(f2,[1],ones(3)'/2, color=co[3:5]', 
            label=[ L"x_1" L"x_2" L"x_3"], legendfont=16, leg=:outerright)
    #plot!(f3,[1],zeros(3)', color=co[3:5]', 
    #        label=[L"q_1" L"q_3" L"q_3"], legendfont=16, leg=:outerright)
    plot!(f3,[1],zeros(1), color=co[3],label=false,  
            ylabel=L"\bar{c}(t)", legendfont=16, leg=:outerright)
    plot!(f3,[0;T], [1;1].*mos.game_param.c_star, color=:black, xlabel="t",
            lw=2 ,ls=:dash, label=false, legendfont=16, leg=:outerright)




    plot!(
        f1,f2,f3, 
        layout=@layout [°; °; °]
    )
end



#%%

@show mos.game_param.k
f = do_sim(mos,z0)
savefig(f, "EPG_many_learning_rules_well_controlled.png")

mos2 = copy(mos)
@show mos2.game_param.k = [1.0;1.0;1.0]
f = do_sim(mos2,z0)
savefig(f, "EPG_many_learning_rules_poorly_controlled.png")



#%%

