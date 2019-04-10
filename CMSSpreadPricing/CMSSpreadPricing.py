import math
import numpy as np
import numba as nb
from scipy.stats import norm
from scipy.integrate import quadrature, quad, fixed_quad
from scipy import LowLevelCallable
import xlwings as xw

##################################################
############## CMS CONVEXITY COMPUTATION #########
##################################################

@xw.func
def cms_convexity(swap_forward_rate: float, ois_forward_rate: float, 
                    forward_swap_volatility: float, ois_forward_swap_volatility: float,
                    libor_ois_correlation: float, time_to_fixing: float,
                    swap_tenor_years: float, fixing_frequency: float,
                    payment_frequency: float) -> float:
    """
    Returns CMS Convexity adjustment in presence of LIBOR - OIS spread (Multiple Curves framework).
    """
    k_correction = CMS_k_function(swap_forward_rate, ois_forward_rate, 
                                    fixing_frequency, payment_frequency,
                                    swap_tenor_years)
    factor1 = math.exp((forward_swap_volatility**2)*time_to_fixing) - 1
    factor2 = math.exp(libor_ois_correlation*forward_swap_volatility*ois_forward_swap_volatility*time_to_fixing) - 1 
    return k_correction * (factor1 - factor2*(swap_forward_rate - ois_forward_rate)/swap_forward_rate)

@xw.func
def forward_swap_cms_correction(swap_forward_rate: float, cms_forward_rate: float,
                                time_to_fixing: float) -> float:
    """
    Returns the exponential factor correction for the convexity induced by taking
    the expected value of the future swap under the measure induced by the 
    fixing-date maturity zero coupon bond.
    """
    frac = cms_forward_rate / swap_forward_rate
    return math.log(frac) / time_to_fixing

@xw.func
def spread_option_price_murex(cms_forward_1: float, cms_forward_2: float, 
                                vol_1: float, vol_2: float,
                                correlation: float, time_to_maturity: float,
                                strike: float, option_type: str, 
                                deflator: float=1) -> float:
    """
    Computes the undiscounted value of CMS Spread call option based
    on Murex documentation 1088.12.7  
    """
    spread_pricer = CMSSpreadPricer_murex(cms_forward_1, cms_forward_2,
                                            vol_1, vol_2,
                                            time_to_maturity,
                                            option_type)
    return spread_pricer.option_price(correlation, strike, deflator)

@xw.func
def spread_option_price_brigo_mercurio(forward_1: float, forward_2: float, 
                                        vol_1: float, vol_2: float,
                                        mu_1: float, mu_2: float,
                                        correlation: float, time_to_maturity: float,
                                        strike: float, option_type: str,
                                        deflator: float=1) -> float:
    """
    Computes the undiscounted value of a CMS Spread option based 
    on Brigo and Mercurio CMSSO formula (pag 604).
    """
    spread_pricer = CMSSpreadPricer_bm(forward_1, forward_2,
                                        vol_1, vol_2,
                                        mu_1, mu_2,
                                        time_to_maturity,
                                        option_type)
    return spread_pricer.option_price(correlation, strike, deflator)

@xw.func
def spread_option_normal_model(cms_spread: float, bp_volatility: float,
                                time_to_maturity: float, strike: float,
                                option_type: float, deflator: float=1) -> float:
    """
    Computes the undiscounted value of a CMS Spread option assuming that
    the CMS Spread is a normally distributed random variable.
    """
    std_dev = bp_volatility * math.sqrt(time_to_maturity)

    d = cms_spread - strike
    d /= std_dev

    Nd = norm.cdf(d)
    nd = std_dev * norm.pdf(d)

    intrinsic_value = cms_spread - strike
    if option_type.lower() == 'cap':
        return deflator*(intrinsic_value*Nd + nd)
    elif option_type.lower() == 'floor':
        return deflator*(intrinsic_value*(Nd - 1) + nd)
    else:
        error_message = '{} is not a valid option type'.format(option_type)
        raise ValueError(error_message)


cSignature = nb.types.double(nb.types.intc, nb.types.CPointer(nb.types.double))
@nb.cfunc(cSignature)
def integrand_function_brigo_mercurio(n, xx):
    in_array = nb.carray(xx, (n,))
    v, fwd_1, fwd_2, mu_1, mu_2, sigma_1, sigma_2, rho, tau, strike = in_array

    h_exponent = (mu_2 - 0.5*sigma_2*sigma_2)*tau
    h_exponent += sigma_2*math.sqrt(tau) * v
    h = strike + fwd_2*math.exp(h_exponent)

    k_exponent = mu_1*tau - 0.5*rho*rho*sigma_1*sigma_1*tau 
    k_exponent += rho*sigma_1*math.sqrt(tau)*v
    k = fwd_1*math.exp(k_exponent)

    d1_factor = mu_1 + sigma_1*sigma_1*(0.5 - rho*rho)
    d2_factor = mu_1 - 0.5*sigma_1*sigma_1

    num_factor = rho*sigma_1*math.sqrt(tau)*v
    moneyness_h = math.log(fwd_1/h)
    den = sigma_1 * math.sqrt(1 - rho*rho) * math.sqrt(tau)

    d1 = moneyness_h + d1_factor*tau + num_factor
    d1 /= den

    d2 = moneyness_h + d2_factor*tau + num_factor
    d2 /= den

    N_d1 = 0.5*(1 + math.erf(d1/math.sqrt(2)))
    N_d2 = 0.5*(1 + math.erf(d2/math.sqrt(2)))

    n_pdf = math.exp(-0.5*v*v)
    n_pdf /= math.sqrt(2*math.pi)

    return n_pdf*(k*N_d1 - h*N_d2)

cms_integrand_brigo_mercurio = LowLevelCallable(integrand_function_brigo_mercurio.ctypes)

@nb.cfunc(cSignature)
def integrand_function_murex_docs(n, xx):
    in_array = nb.carray(xx, (n,))
    v, fwd_1, fwd_2, sigma_1, sigma_2, rho, tau, strike, w = in_array

    sigma_1 = sigma_1*math.sqrt(tau)
    sigma_2 = sigma_2*math.sqrt(tau)

    mu_1 = -0.5*sigma_1*sigma_1
    mu_2 = -0.5*sigma_2*sigma_2

    h_exponent = mu_1 + sigma_1*v
    h = strike + fwd_1*math.exp(h_exponent)

    k_exponent = mu_2 + rho*sigma_2*v
    k_exponent += 0.5*sigma_2*sigma_2*(1 - rho*rho)
    k = fwd_2*math.exp(k_exponent)

    if h <= 0:
        f = 0.5*(1 - w)*(k - h)
    else:
        d2_factor = mu_2 + rho*sigma_2*v
        d1_factor = d2_factor + sigma_2*sigma_2*(1 - rho*rho)

        moneyness_h = math.log(fwd_2/h)
        den = sigma_2*math.sqrt(1 - rho*rho)

        d1 = moneyness_h + d1_factor
        d1 /= den

        d2 = moneyness_h + d2_factor
        d2 /= den

        N_d1 = 0.5*(1 + math.erf(-w*d1/math.sqrt(2)))
        N_d2 = 0.5*(1 + math.erf(-w*d2/math.sqrt(2)))

        f = w*h*N_d2 -w*k*N_d1

    n_pdf = math.exp(-0.5*v*v)
    n_pdf /= math.sqrt(2*math.pi)

    return n_pdf*f

cms_integrand_murex = LowLevelCallable(integrand_function_murex_docs.ctypes)


##################################################
############## CMS SPREAD OPTION #################
##################################################

class CMSSpreadPricer_bm:
    def __init__(self, swap_forward_rate_1: float, swap_forward_rate_2: float,
                    swap_volatility_1: float, swap_volatility_2: float,
                    swap_convexity_1: float, swap_convexity_2: float,
                    time_to_fixing: float, option_type: str) -> None:

        if option_type.lower() == 'cap':
            self.w = 1
        elif option_type.lower() == 'floor':
            self.w = -1 
        else: 
            error_message = '{} is not a valid option type'.format(option_type)
            raise ValueError(error_message)
       
        self.fwd_1 = swap_forward_rate_1
        self.fwd_2 = swap_forward_rate_2

        self.mu_1 = swap_convexity_1
        self.mu_2 = swap_convexity_2
        self.vol_1 = swap_volatility_1
        self.vol_2 = swap_volatility_2

        self.tau = time_to_fixing

    def option_price(self, p: float, k: float, deflator: float=1) -> float:
        if self.w == 1:
            return deflator*self.call_price(p, k)
        else:
            cms_1 = self.fwd_1*math.exp(self.mu_1 * self.tau)
            cms_2 = self.fwd_2*math.exp(self.mu_2 * self.tau)
            forward_price = cms_1 - cms_2 - k
            call_price = self.call_price(p, k)
            return deflator*(call_price - forward_price)

    def call_price(self, p: float, k: float) -> float:
        data = (self.fwd_1, self.fwd_2, self.mu_1, self.mu_2,
                self.vol_1, self.vol_2, p, self.tau, k)
        return quad(cms_integrand_brigo_mercurio, a=-7, b=7, args=data)[0]


class CMSSpreadPricer_murex:
    def __init__(self, cms_forward_1: float, cms_forward_2:float,
                    cms_volatility_1: float, cms_volatility_2: float,
                    time_to_fixing: float,
                    option_type: str) -> None: 
        if option_type.lower() == 'cap':
            self.w = 1
        elif option_type.lower() == 'floor':
            self.w = -1 
        else: 
            error_message = '{} is not a valid option type'.format(option_type)
            raise ValueError(error_message)
        
        self.cms_fwd_1 = cms_forward_1
        self.cms_fwd_2 = cms_forward_2
        
        self.cms_vol_1 = cms_volatility_1
        self.cms_vol_2 = cms_volatility_2

        self.tau = time_to_fixing
    
    def option_price(self, p: float, k: float, deflator: float=1) -> float:
        data = (self.cms_fwd_1, self.cms_fwd_2, 
                self.cms_vol_1, self.cms_vol_2,
                p, self.tau, -k, self.w)
        return deflator*quad(cms_integrand_murex, a=-7, b=7, args=data)[0]
        

def CMS_k_function(swap_rate: float, ois_rate: float, 
                    fixing_frequency: float, payment_frequency: float,
                    swap_tenor_years: float) -> float:
    k_rt = 1 + (fixing_frequency-payment_frequency)*ois_rate
    k_rt -= swap_tenor_years*fixing_frequency*ois_rate/((1 + fixing_frequency*ois_rate)**swap_tenor_years - 1)
    k_rt *= (swap_rate**2)/(ois_rate * (1 + fixing_frequency*ois_rate))
    return k_rt
 
