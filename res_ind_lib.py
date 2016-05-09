import numpy as np
import pandas as pd
from scipy.special import erf
from scipy.interpolate import interp1d
   

def compute_resiliences(df_in,kind="exante",fa_ratios=None, is_relative_to_local=True,verbose_output=True):
    """Main function. Computes all outputs (dK, resilience, dC, etc,.) from inputs"""
    
    #keeps tracks of the variables that need to be integrated over rps
    
    if fa_ratios is not None:

        #grid of rps consistent with fa_ratios and protection
        fa_ratios = interpolate_faratios(fa_ratios,df_in.protection.unique().tolist())

        #builds a dataframe multi-indexed by return period  (country, (var, rp))
        nrps =len(fa_ratios.columns)
        df = pd.concat(
            [df_in.copy(deep=True).select_dtypes(exclude=[object])]*nrps,
            axis=1, keys=fa_ratios.columns, names=["rp","var"]
            ).swaplevel("var","rp",axis=1).sortlevel(0,axis=1)
        
        #introduces different exposures for different return periods
        df["fa"]=df["fa"]*fa_ratios
        
        #Reshapes into ((country,rp), vars) formally using only country as index
        df=df.stack("rp").reset_index().set_index("country")
        
    else:
        df=df_in.copy(deep=True)
    
    # # # # # # # # # # # # # # # # # # #
    # MACRO
    # # # # # # # # # # # # # # # # # # #
    
    #exogenous discount rate
    rho= 5/100
    
    #productivity of capital
    mu= df["avg_prod_k"]

    #rebuilding exponentially to 95% of initial stock in reconst_duration
    three = np.log(1/0.05) 
    recons_rate = three/ df["T_rebuild_K"]  
    
    # Calculation of macroeconomic resilience
    gamma =(df["alpha"]*mu +recons_rate)/(rho+recons_rate)  
    
    if verbose_output: 
        df["macro_multiplier"]=gamma
   
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
    #Eta
    elast=  df["income_elast"]

    if verbose_output: 
        df["cp"]=cp
        df["cr"]=cr
    
    ###########"
    #vulnerabilities from total and bias
    if kind=="exante":
        #early-warning-adjusted vulnerability
        v_shew=df["v"]*(1-df["pi"]*df["shew"])
        vs_shew = df["v_s"] *(1-df["pi"]*df["shew"])
        
    else:
        v_shew = df["v"]
        vs_shew = df["v_s"]
    
    df["v_shew"]=v_shew    
    
    pv=df.pv
    
    #poor and non poor vulnerability "protected" from exposure variations... 
    vp_shew, vr_shew= unpack_v(v_shew,pv,df.faref,df.peref,ph,df.share1_ref)
    
    df["v_p"]=vp_shew
    df["v_r"]=vr_shew
    
    if kind=="exante":
        # Ability and willingness to improve transfers after the disaster
        df["borrow_abi"]=(df["rating"]+df["finance_pre"])/2 
        sig=.50*(df["borrow_abi"]+df["prepare_scaleup"])/2
        df["sigma_p"]=sig 
        df["sigma_r"]=sig 
        

    tot_p=1-(1-df["social_p"])*(1-df["sigma_p"])
    tot_r=1-(1-df["social_r"])*(1-df["sigma_r"])
    
    if verbose_output:
        df["tot_p"]=tot_p
        df["tot_r"]=tot_r

    share_shareable = 1 if kind=="exante" else df["share_nat_income"]
        
    #No protection for ex post analysis
    # protection = pd.Series(index=df.index).fillna(0) if kind=="expost" else df["protection"]
    
    ############################
    #### Parameters for poverty traps
    
    #cost of poverty trap
    three = np.log(1/0.05) 
    recover_rate= three/df["T_rebuild_L"] 
    df["dC_destitution"]= df["gdp_pc_pp"]/(rho+recover_rate)

    #social exposure to poverty traps
    df["institutional_exposure"]=((1-df["axhealth"])+(1-df["plgp"])+df["unemp"])/3   
    
    ####################################""""
    #### EXPOSURE
    
    fa=df.fa 
    
    #exposures from total exposure and bias
    pe=df.pe
    fap =(fa*(1+pe))
    far =((fa-ph*fap)/(1-ph))
    
    if verbose_output:
        df["fap"]= fap
        df["far"]= far
    
    ######################################
    #Welfare losses from consumption losses
    
    delta_W,dK,foo,foo    =calc_delta_welfare(ph,fap,far,vp_shew,vr_shew,vs_shew,cp,cr,tot_p,tot_r,mu,share_shareable,gamma,rho,elast)
    
    if verbose_output:
        #Assuming no scale up
        dW_noupscale    ,foo,foo,foo  =calc_delta_welfare(ph,fap,far,vp_shew,vr_shew,vs_shew,cp,cr,df["social_p"],df["social_r"],mu,share_shareable,gamma,rho,elast)

        # Assuming no transfers at all
        dW_no_transfers   ,foo,foo,foo  =calc_delta_welfare(ph,fap,far,vp_shew,vr_shew,vs_shew,cp,cr,0             ,0             ,mu,share_shareable,gamma,rho,elast)
        
        df["dW_noupscale"]= dW_noupscale/df["protection"]
        df["dW_no_transfers"]= dW_no_transfers/df["protection"]
        # extensive_vars += ["dW_noupscale","dW_no_transfers"]

    del foo    
        
    #output
    df["delta_W"]= delta_W/df["protection"]
    df["dKpc"]=dK/df["protection"]
    df["dKtot"]=dK*df["pop"]/df["protection"]
    
    df["risk_to_assets"] = dK/df.gdp_pc_pp/df.protection
    # extensive_vars += ["delta_W","dKpc","dKtot"]
    
    #######################################
    #Welfare losses from poverty traps

    
    #fraction of people with very low income
    trap_treshold = 0.1*df["gdp_pc_pp"]
    
    individual_exposure =(
            ph*fap*(1-df["axfin_p"])*THETA(cp-trap_treshold,cp*(1-tot_p)*vp_shew,df["H"])+
        (1-ph)*far*(1-df["axfin_r"])*THETA(cr-trap_treshold,cr*(1-tot_r)*vr_shew,df["H"])
    )
    
    del trap_treshold, fap, far
    
    #total exposure
    tot_exposure =  individual_exposure *  df["institutional_exposure"]
    
    df["dW_destitution"]=(tot_exposure*(welf(df["gdp_pc_pp"]/rho,elast) - welf(df["gdp_pc_pp"]/rho-df["dC_destitution"],elast)))/df["protection"]
    
    # extensive_vars+=["dW_destitution"]
    
    if verbose_output:
        df["destitution_exposure"]= tot_exposure
        df["individual_exposure"]= individual_exposure
     
    ###AGGREGATION OF THE OUTPUTS OVER RETURN PERIODS
    if fa_ratios is not None:

        #computes probability of each return period
        
        fa_ratios.drop([0],axis=1, errors ="ignore", inplace=True)
        
        
        proba = pd.Series(np.diff(np.append(1/fa_ratios.columns.values,0)[::-1])[::-1],index=fa_ratios.columns) #removes 0 from the rps
        
        #matches return periods and their probability
        proba_serie=df.rp.replace(proba)
        
        #removes events below the protection level
        proba_serie[df.protectionref>df.rp] =0
        
        #average weighted by proba
        f=df.mul(proba_serie,axis=0).sum(level="country").div(proba_serie.sum(level="country"),axis=0)
        
        # f[extensive_vars]=f[extensive_vars].mul(proba_serie.sum(level="country"),axis=0)
        
        df = df_in.copy(deep=True)
        df[f.columns]=f
        # df[f.columns].fillna(0) #Those countries protected at max
        
        del f
        
    
        # df[extensive_vars]=df[extensive_vars].div(df["protection"],axis=0)
    
    ##########""
    # CMPUTES RISK AND RESILIENCE AS RATIOS OF VARIABLES AERAGES OVER RPs
    # (make sure variables are accessed from df, not from convenience local variables)
    
    ############################
    #Reference losses
    h=1e-4
    
    if is_relative_to_local:
        wprime =(welf(df["gdp_pc_pp"]/rho+h,df["income_elast"])-welf(df["gdp_pc_pp"]/rho-h,df["income_elast"]))/(2*h)
    else:
        wprime = (welf(df["gdp_pc_world"]/rho+h,df["income_elast"])-welf(df["gdp_pc_world"]/rho-h,df["income_elast"]))/(2*h)
    
    ###################"
    #Risk
    
    df["equivalent_cost"]=(df["dW_destitution"]+df["delta_W"])/(wprime )
    df["risk"]=df["equivalent_cost"]/(df["gdp_pc_pp"])
    
    if verbose_output:
        df["dWsurWprime"]=df["delta_W"]/wprime
        df["total_equivalent_cost"]=(df["equivalent_cost"]*df["pop"])
        df["total_equivalent_cost_of_destitution"]=df["total_equivalent_cost"]*df["dW_destitution"]/(df["dW_destitution"]+df["delta_W"])

        df["total_equivalent_cost_no_destitution"]=df["total_equivalent_cost"]*        df["delta_W"]/(df["dW_destitution"]+df["delta_W"])

    
    ############################
    #Resilience
    df["resilience"]=df["dKpc"]/(df["equivalent_cost"] )
    
    if verbose_output:
        dWref   = wprime*df["dKpc"] 
        df["resilience_no_shock"]=dWref/df["delta_W"]                
        df["resilience_no_shock_no_uspcale"]=dWref/df["dW_noupscale"]          
        df["resilience_no_shock_no_SP"]=dWref/df["dW_no_transfers"]       

        #orders columns alphanumerically
        df=df.reindex_axis(sorted_nicely(df.columns), axis=1)

    return df

# TODO : AVOID THIS "LOOP" AND RELY ON ASSIGN outside the main function to compute this
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
    dK = kp*vp*nap+\
         kr*vr*nar
    
    # consumption losses per category of population
    d_cur_cnp=fap*v_shared *la_p *kp *sh_sh   #v_shared does not change with v_rich so pv changes only vulnerabilities in the affected zone. 
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
    
    y=(c**(1-elast)-1)/(1-elast)
    
    #log welfare func
    cond = (elast==1)
    y[cond] = np.log(c[cond]) 
    
    return y
        
    
def THETA(y,m,H):
    """P(x>y) where x follows log-normal of average=m and Homogeneity=H"""
    #h=med/m = exp(-s**2/2)   log(h) = -s**2/2
    #h=med/m   med = m*h = exp(mu)   mu = log(m*h)
    return .5 * (1-erf(np.log(y/(m*H))/(2*np.sqrt(-np.log(H)))))
   

def unpack_v(v,pv,fa,pe,ph,share1):
#poor and non poor vulnerability from aggregate vulnerability,  exposure and biases
    
    v_p = v*(1+pv)
    
    fap_ref= fa*(1+pe)
    far_ref=(fa-ph*fap_ref)/(1-ph)
    cp_ref=   share1 /ph
    cr_ref=(1-share1)/(1-ph)
    
    x=ph*cp_ref *fap_ref    
    y=(1-ph)*cr_ref  *far_ref
    
    v_r = ((x+y)*v - x* v_p)/y
    
    return v_p,v_r

def compute_v_fa(df):
    fap = df["fap"]
    far = df["far"]
    
    vp = df.v_p
    vr=df.v_r

    ph = df["pov_head"]
        
    cp=df["share1"]/ph 
    cr=(1-df["share1"])/(1-ph)
    
    fa = ph*fap+(1-ph)*far
    
    x=ph*cp 
    y=(1-ph)*cr 
    
    v=(y*vr+x*vp)/(x+y)
    
    return v,fa
    
    
def def_ref_values(df,kind="exante"):
    #fills the "ref" variables (those protected when computing derivatives)
    df["peref"]=df["pe"]
    df["faref"]=df["fa"]
    if kind=="exante":
        df["protectionref"]=df["protection"]
    else:
        df["protectionref"]=df["protection"]=1
        
    df["share1_ref"]=df["share1"]
    vp,vr =unpack_v(df.v,df.pv,df.fa,df.pe,df.pov_head,df.share1)
    df["v_s"] = vr
    return df
    
    
    
def interpolate_faratios(fa_ratios,protection_list):
 
    #figures out all the return periods to be included
    all_rps = list(set(protection_list+fa_ratios.columns.tolist()))

    fa_ratios_rps = fa_ratios.copy()
    
    #extrapolates linear towards the 0 return period exposure  (this creates negative exposure that is tackled after interp) (mind the 0 rp when computing probas)
    fa_ratios_rps[0]=fa_ratios_rps.iloc[:,0]- fa_ratios_rps.columns[0]*(
        fa_ratios_rps.iloc[:,1]-fa_ratios_rps.iloc[:,0])/(
        fa_ratios_rps.columns[1]-fa_ratios_rps.columns[0])
    
    
    #add new, interpolated values for fa, assuming constant exposure on the right
    x = fa_ratios_rps.columns.values
    y = fa_ratios_rps.values
    fa_ratios_rps= pd.concat(
        [pd.DataFrame(interp1d(x,y,bounds_error=False)(all_rps),index=fa_ratios_rps.index, columns=all_rps)]
        ,axis=1).sort_index(axis=1).clip(lower=0).fillna(method="pad",axis=1)
    fa_ratios_rps.columns.name="rp"

    return fa_ratios_rps
    
import itertools
from progress_reporter import *    
def compute_derivative(df_original,score_card_set,deriv_set, fa_ratios=None, **kwargs):
        
    # kwargs.update(verbose_output=True)
        
    #makes sure v_shew is in score_card_set    
    score_card_set = np.unique((score_card_set+["v_shew"])).tolist()    
        
    #der = pd.DataFrame()
    h=2e-3
    
    if fa_ratios is not None:
        #grid of rps consistent with fa_ratios and protection
        fa_ratios = interpolate_faratios(
            fa_ratios,(
                df_original.protection.unique().tolist())+
                (h+df_original.protection).unique().tolist()
                                        )
        kwargs.update(fa_ratios=fa_ratios)

    #loop on all data in df prior to add the results
    fx = compute_resiliences(df_original,  **kwargs)[score_card_set]

    headr = list(itertools.product(deriv_set,score_card_set))
    #(countries, (input vars, outpus))
    der=  pd.DataFrame(index=df_original.dropna().index, columns=pd.MultiIndex.from_tuples(headr,names=["var","out"])).sortlevel(0,axis=1) 
    #der.columns
    for var in deriv_set:
        progress_reporter(var)
        df_h=df_original.copy(deep=True)
        df_h[var]=df_h[var]+h
        fxh= compute_resiliences(df_h, **kwargs)[score_card_set]
        der[var] = (fxh-fx)/(h)  #this reads (all countries, (this var, all output))  = (all countries, all output)

    
    der=der.swaplevel("var","out",axis=1).sortlevel(0,axis=1) 

    #derivatives of risk with respect to resilience
    der_risk_rel=  pd.DataFrame((der["risk"]["axfin_p"]/der["resilience"]["axfin_p"]))
      
    der_risk_rel.columns = pd.MultiIndex.from_product(["risk", "resilience"])

    
    #derivatives of risk with respect to vulnerability
    der_risk_vshew = pd.DataFrame((der["risk"].v/der["v_shew"].v))
    der_risk_vshew.columns = pd.MultiIndex.from_product(["risk", "v_shew"])

    derivatives =     pd.concat([der,der_risk_rel,der_risk_vshew], axis=1).sortlevel(0,axis=1)# (countries, (outputs, inputs))


    #Signs of resilience derivative 
    der = np.sign(derivatives["resilience"]).replace(0,np.nan)
    signs= pd.Series(index=der.columns)
    for i in signs.index:
        if (der[i].min()==der[i].max()): #all nonnan signs are equal
            signs[i]=der[i].min()
        else:
            print("ambigous sign for "+i)
            signs[i]=np.nan
            
    return derivatives       
    
    
import re
def sorted_nicely(l): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)        