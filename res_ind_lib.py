import numpy as np
import pandas as pd
from scipy.special import erf
import matplotlib.pyplot as plt

 
def compute_resiliences(df_in,kind="exante"):
    """Mains function. Computes all outputs (dK, resilience, dC, etc,.) from inputs"""
    
    df=df_in.copy(deep=True)

    # # # # # # # # # # # # # # # # # # #
    # MACRO
    # # # # # # # # # # # # # # # # # # #
    
    # Effect of production and reconstruction
    ripple_effects=  1+df["alpha"] 
    
    #rebuilding exponentially to 95% of initial stock in reconst_duration
    three = np.log(1/0.05) 
    recons_rate = three/ df["T_rebuild_K"]  
    
    #exogenous discount rate
    rho= 5/100
    
    #productivity of capital
    mu= df["avg_prod_k"]

    # Calculation of macroeconomic resilience
    df["macro_multiplier"]=gamma =(ripple_effects*mu +recons_rate)/(rho+recons_rate)   
   

    # # # # # # # # # # # # # # # # # # #
    # MICRO 
    # # # # # # # # # # # # # # # # # # #
    
    
    ###############################
    #Description of inequalities
    
    #proportion of poor people
    ph = df["pov_head"]
    
    #consumption levels
    cp=   df["share1"] /ph    *df["gdp_pc_pp"]
    cr=(1-df["share1"])/(1-ph)*df["gdp_pc_pp"]
    C = df["gdp_pc_pp"]

    #exposures from total exposure and bias
    fa=df.fa
    pe=df.pe
    fap =df["fap"]=fa*(1+pe)
    far =df["far"]=(fa-ph*df.fap)/(1-ph)

    #vulnerabilities from total and bias
    if kind=="exante":
        #early-warning-adjusted vulnerability
        v_shew=df["v"]*(1-df["pi"]*df["shew"])
        vs_shew = df["v_s"] *(1-df["pi"]*df["shew"])
        
    else:
        v_shew = df["v"]
        vs_shew = df["v_s"]
        
    df["v_shew"] = v_shew
    pv=df.pv
    df["v_p"] = v_shew*(1+pv)
    
    #non poor vulnerability "protected" from exposure variations... 
    fap_ref= df.faref*(1+df.peref)
    far_ref=(df.faref-ph*fap_ref)/(1-ph)
    cp_ref=   df["share1_ref"] /ph
    cr_ref=(1-df["share1_ref"])/(1-ph)
    
    x=ph*cp_ref *fap_ref    
    y=(1-ph)*cr_ref  *far_ref
    
    df["v_r"] = ((x+y)*v_shew - x* df["v_p"])/y

    vp_shew = df["v_p"]
    vr_shew = df["v_r"]
  
    

    if kind=="exante":
        # Ability and willingness to improve transfers after the disaster
        df["borrow_abi"]=(df["rating"]+df["finance_pre"])/2 

        df["sigma_p"]=.50*(df["borrow_abi"]+df["prepare_scaleup"])/2
        df["sigma_r"]=.50*(df["borrow_abi"]+df["prepare_scaleup"])/2

    df["tot_p"]=tot_p=1-(1-df["social_p"])*(1-df["sigma_p"])
    df["tot_r"]=tot_r=1-(1-df["social_r"])*(1-df["sigma_r"])
    
    if kind=="exante":
        share_shareable = 1
    else:
        share_shareable = df["share_nat_income"]
        
    ######################################
    #Welfare losses from consumption losses
    
    #Eta
    elast=  df["income_elast"]

    deltaW,dK,df["dcap"],df["dcar"]    =calc_delta_welfare(ph,fap,far,vp_shew,vr_shew,vs_shew,cp,cr,tot_p         ,tot_r,mu,share_shareable,gamma,rho,elast)
    
    df["delta_W"]=deltaW
    df["dKpc"]=dK
    
    df["dKtot"]=dK*df["pop"]
    df["cp"]=cp
    df["cr"]=cr
    
    #Assuming no scale up
    dW_noupscale    ,foo,df["dcapns"],df["dcarnos"]    =calc_delta_welfare(ph,fap,far,vp_shew,vr_shew,vs_shew,cp,cr,df["social_p"],df["social_r"],mu,share_shareable,gamma,rho,elast)

    # Assuming no transfers at all
    dW_no_transfers   ,foo,foo,foo  =calc_delta_welfare(ph,fap,far,vp_shew,vr_shew,vs_shew,cp,cr,0             ,0             ,mu,share_shareable,gamma,rho,elast)
    
    #######################################
    #Welfare losses from poverty traps
    
    #social exposure to poverty traps
    df["institutional_exposure"] =institutional_exposure= ((1-df["axhealth"])+(1-df["plgp"])+df["unemp"])/3   
    
    #individual exposure
    psi_p=df["axfin_p"]
    psi_r=df["axfin_r"]
    
    #fraction of people with very low income
    trap_treshold = 0.1*df["gdp_pc_pp"]
    
    df["individual_exposure"]= individual_exposure =\
            ph*fap*(1-psi_p)*THETA(cp-trap_treshold,cp*(1-tot_p)*vp_shew,df["H"])+\
        (1-ph)*far*(1-psi_r)*THETA(cr-trap_treshold,cr*(1-tot_r)*vr_shew,df["H"])
          
    #total exposure
    df["destitution_exposure"] =tot_exposure =  individual_exposure *  institutional_exposure

    #cost of poverty trap
    recover_rate= three/df["T_rebuild_L"] 
    df["dC_destitution"]=dC_destitution =df["gdp_pc_pp"]/(rho+recover_rate)

    
    df["dW_destitution"]= dW_destitution =tot_exposure*(welf(C/rho,elast) - welf(C/rho-dC_destitution,elast))
     
    ############################
    #Reference losses
    h=1e-4
    wprime =(welf(df["gdp_pc_pp"]/rho+h,elast)-welf(df["gdp_pc_pp"]/rho-h,elast))/(2*h)
    dWref   = wprime*dK
    
    ############################
    #Risk and resilience
    
    #resilience
    df["resilience"]=dWref/(dW_destitution+deltaW);
    df["resilience_no_shock"]=dWref/deltaW;
    df["resilience_no_shock_no_uspcale"]=dWref/dW_noupscale;
    df["resilience_no_shock_no_SP"]=dWref/dW_no_transfers;
       
    #Probability that the event happens
    if kind=="exante":
        proba = 1/df["protection"]
    else:
        proba = 1
    
    #Risk
    
    df["dWsurWprime"]=deltaW/wprime
    
    df["equivalent_cost"] = proba* (dW_destitution+deltaW)/wprime
    
    df["risk"]= df["equivalent_cost"]/(df["gdp_pc_pp"]);
    
    df["total_equivalent_cost"]=df["equivalent_cost"]*df["pop"];
    df["total_equivalent_cost_of_destitution"]=df["total_equivalent_cost"]*dW_destitution/(dW_destitution+deltaW)
    df["total_equivalent_cost_no_destitution"]=df["total_equivalent_cost"]*        deltaW/(dW_destitution+deltaW)
  
    return df

    
    
def calc_delta_welfare(ph,fap,far,vp,vr,v_shared,cp,cr,la_p,la_r,mu,sh_sh,gamma,rho,elast):
    """welfare cost from consumption losses"""

    #fractions of people non-poor/poor affected/non affected over total pop
    nap= ph*fap
    nar=(1-ph)*far
    nnp=ph*(1-fap )
    nnr=(1-ph)*(1-far)
    
    #capital from consumption and productivity
    kp = cp/mu
    kr = cr/mu
    
    #total capital losses
    dK = kp*vp*fap*ph+\
         kr*vr*far*(1-ph)
    
    # consumption losses per category of population
    d_cur_cnp=fap*v_shared *la_p* kp *sh_sh   #v_shared does not change with v_rich so pv changes only vulnerabilities in the affected zone. 
    d_cur_cnr=far*v_shared *la_r *kr *sh_sh
    d_cur_cap=vp*(1-la_p)*kp + d_cur_cnp
    d_cur_car=vr*(1-la_r)*kr + d_cur_cnr
    
    #losses in NPV after reconstruction 
    d_npv_cnp= gamma*d_cur_cnp
    d_npv_cnr= gamma*d_cur_cnr
    d_npv_cap= gamma*d_cur_cap
    d_npv_car= gamma*d_cur_car
    
    #welfare cost 
    Wpre =ph* welf(cp/rho,elast) + (1-ph)*welf(cr/rho,elast)
    Wpost=  nap*welf(cp/rho-d_npv_cap,elast) + \
            nnp*welf(cp/rho-d_npv_cnp,elast) + \
            nar*welf(cr/rho-d_npv_car,elast)+ \
            nnr*welf(cr/rho-d_npv_cnr,elast)
    dW =Wpre -Wpost #counting losses as +

    return dW,dK, d_cur_cap, d_cur_car
    
   
def welf(c,elast):
    """"Welfare function"""
    
    scale=1e4 #for numerical precision
    
    y=scale*(c**(1-elast)-1)/(1-elast)
    
    cond = elast==1
    y[cond] = scale*np.log(c[cond]) 
    
    return y
        
    
def THETA(y,m,H):
    """P(x>y) where x follows log-normal of average=m and Homogeneity=H"""
    #h=med/m = exp(-s**2/2)   log(h) = -s**2/2
    #h=med/m   med = m*h = exp(mu)   mu = log(m*h)
    return .5 * (1-erf(np.log(y/(m*H))/(2*np.sqrt(-np.log(H)))))
    
def compute_v_fa(df):
    fap = df["fap"]
    far = df["far"]
    
    vp = df.v_p
    vr=df.v_r

    ph = df["pov_head"]
        
    cp=df["share1"]/ph*df["gdp_pc_pp"]
    cr=(1-df["share1"])/(1-ph)*df["gdp_pc_pp"]
    
    fa = ph*fap+(1-ph)*far
    
    x=ph*cp 
    y=(1-ph)*cr 
    
    v=(y*vr+x*vp)/(x+y)
    
    return v,fa
    