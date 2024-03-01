include("lib/Aux.jl")
include("lib/Protocols.jl")
include("lib/Dynamics.jl")
include("bound_I.jl")
include("bound_payoff.jl")

#%%

#select target equilibria, and update the System Description
function pick_target_equilibrium(sd::SystemDescription)
    B = sd.cs_param.parameters[1]
    c = sd.game_param.c
    c_star = sd.game_param.c_star
    
    
    m = Model(Ipopt.Optimizer)
    set_attribute(m, "print_level", 0)
    BB(x...) = B(collect(x))
    register(m, :BB, mos.game_param.ns, BB; autodiff = true)

    @variable(m, x[1:mos.game_param.ns]>=0)
    @constraint(m, sum(x)==1)
    @constraint(m, c'*x<=c_star)
    @NLobjective(m, Min, BB(x...))
    
    optimize!(m)

    value.(x)
end

#payoff dynamics
function pdm(sd::SystemDescription,Z,t)
    local y,x,p = Z
    local I,R = y
    
    local x_star = sd.game_param.x_star
    local B = sd.cs_param.parameters[1]
    local k1,k2,k3 = sd.game_param.k
    local p_star = sd.game_param.p_star

    a(x) = B(x)/(γ+δ*Rb(sd,B(x)))
    U(x) = (I-Ib(sd,B(x)))+Ib(sd,B(x))*log(Ib(sd,B(x))/I)+a(x)*(R-Rb(sd,B(x)))^2/2


    if Ib(sd,B(x))/I <= 0 
        #this should never happen! But numeric error could cause it for `Ib` small 
        @show x, I
        @show Ib(sd,B(x))
    end

    dUdx = ForwardDiff.gradient(U,x)

    -k1*dUdx-k2*(x-x_star)-k3*(p-p_star)
end

#payoff function
function F(sd::SystemDescription,Z,t)
    local y,x,p = Z
    p
end

#coupled system dynamics
function sirs(sd::SystemDescription,Z,t)
    local y,x,p =Z
    local B,δ,σ,γ,ω = sd.cs_param.parameters
    local I,R = y
    local S = 1-I-R

    [
        (B(x)*S+δ*I-σ)*I;
        γ*I-ω*R+δ*R*I
    ]
end


## based on our other papers, with some changes
Q = [   0.13  0.18  0.2;
        0.16  0.22  0.23;
        0.17  0.28  0.5]

@assert all( (i<=j && j<k) ? (Q[i,i]<Q[k,k] && Q[i,k]<=Q[j,k] && Q[k,i]<=Q[k,j]) : true 
                    for k=1:3, i=1:3, j=1:3) "Q matrix does not meet assumptions"

function B(x)
      
    x'*Q*x
end

## based on Amaral et al.
#=
function B(x)
    βN = 10.0
    βQ = 1.0
    βa = 0.2*(βQ+βN)/2
    @assert βa < βN
    @assert βQ < βa
    
    
    Q = [βN βa; βa βQ].*0.16 
    x'*Q*x
end
=#




function unconstrained_minB(sd::SystemDescription)
    
    m = Model(Ipopt.Optimizer)

    @variable(m, x[1:3]>=0)
    @constraint(m, sum(x)==1)

    @objective(m, Min, B(x))

    optimize!(m)

    value(B(x))

end



#%%


#same coeffs from previous papers
θ = 0.0002
ζ = 0.0 
δ = 0.005
γ = 0.1
ω = 0.011+θ-ζ 
σ = γ+δ+θ


#mos (my overall system) describes the parameters of the edm+coupled system
mos = SystemDescription(Game(3,3,F,[0.2;0.1;0.0],0.10,zeros(3),zeros(3),[0.1,0.1,0.1],pdm),
                        CoupledSystem(2,sirs,(B,δ,σ,γ,ω))) 

@assert δ < ω
@assert δ < γ
@assert σ < unconstrained_minB(mos)

                        

#%%


mos.game_param.x_star = pick_target_equilibrium(mos)
mos.game_param.p_star = zero(mos.game_param.x_star)

#ϵ = 0.001
#=
for (i,val) in enumerate(mos.game_param.x_star) 
    if val < 1e-9
        mos.game_param.p_star[i] = -ϵ
    else
        mos.game_param.p_star[i] = 0.0
    end
end
=#

βmin = unconstrained_minB(mos)



#we assume that a previous intervention drove the whole population to use
# the safest strategy and the epidemic to reach the lowest endemic 
# equilibrium possible. Now the policy maker will 

Z0 = [ [Ib(mos,βmin);Rb(mos,βmin)], normalize!([1.0;0.0;0.0],1), zeros(3)  ]
z0 = toVec(mos,Z0)


#%%


mos.game_param.k = [1.0; 1.0; 1.0]

Itarget = 0.10


#explore I_max over k_1 and k_2
K1 = 2.0 .*exp.(range(-4,0,length=100))
K2 = 0.04 .*exp.(range(-4,0,length=100)) 



Y = hcat([ begin mos.game_param.k[1]=k1 ; get_I_bound_given_k(mos,2,K2,Z0) end  for k1 in K1]...)

vs = [0.0,0.10, 0.15, 0.23, 0.35, 0.5, 0.8, 1.1]
contourf(K1,K2,map(x->x[1],Y), levels=vs, clabels=true, 
            color=cgrad(:RdYlGn, rev=true), lw=1, xlabel=L"k_1", ylabel=L"k_2", title=L"I_\max"   )

savefig("I_max_over_k1_k2.png")


#%% select k_2
X = 0.1 .*exp.(range(-8,0,length=200)) 
Y = map(x->x[1], get_I_bound_given_k(mos,2,X,Z0) )
plot(X,Y,xscale=:log10)

    ## select it with more resolution
X = range(last(X[Y.<Itarget]),first(X[Y.>Itarget]),length=200) 
Y = map(x->x[1], get_I_bound_given_k(mos,2,X,Z0) )
plot(X,Y)

mos.game_param.k[2] = last(X[Y.<Itarget])
savefig("bound_changing_k2.svg")




#%% select k_1 
X = 10000 .*exp.(range(-10,0,length=200)) 
Y = map(x->x[1], get_I_bound_given_k(mos,1,X,Z0) )
plot(X,Y,xscale=:log10)

    ## select it with more resolution
X = range(last(X[Y.>Itarget]),first(X[Y.<Itarget]),length=200) 
Y = map(x->x[1], get_I_bound_given_k(mos,1,X,Z0) )
plot(X,Y)

mos.game_param.k[1] = (first(X[Y.<Itarget]))
savefig("bound_changing_k1.svg")


# get_p_bound_given_k(mos, Z0) = 0.5486796200662498
@show get_p_bound_given_k(mos,Z0)


@assert all(mos.game_param.k .> 0) "each k_i must be positive"
