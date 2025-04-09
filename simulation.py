import numpy as np
import os
import glob
import pandas as pd

import astropy.units as u
import astropy.constants as c
from astropy.io import fits
import scipy.integrate
from scipy import interpolate
import scipy.stats as st

import matplotlib as mpl
import matplotlib.pyplot as plt
import ast

import orbitize.kepler
from orbitize import cuda_ext, cext

def get_companions(logM, nsamp_Mdot = 1, formpar = 'empirical'):
    """
    simulate (log) mass accretion rate(s) given (log) mass(es) using CASPAR or Stamatellos + Herczeg
    args:
        logM: log mass [M_sun] (single value or np array)
        nsamp_Mdot: number of mass accretion rates to simulate for each mass (default: 1)
        formpar: what formation paradigm to use (empirical or theoretical)
    
    returns:
        logMdot: (array-like) simulated log mass accretion rate(s) [M_sun/yr]
    """
    #choose what linear fit you want
    if formpar =='empirical': a = 2.02; b = -8.92; std = 0.85 #caspar relation (updated 2025)
    elif formpar == 'theoretical':  a = 0.12; b = -10.5; std = 0.3 #S+H relation
    else: print('uh oh! unknown M–Mdot relation :('); return

    #simulate nsamp Mdots for each mass
    if type(logM) == np.ndarray:
        logMdot = np.zeros([len(logM),nsamp_Mdot])
        for i, lM in enumerate(logM):
            lMdot_med = a * lM + b #get Mdot predicted value from linear fit
            offsets = np.random.normal(0, std, nsamp_Mdot) #simulate random scatter
            logMdot[i] = np.random.normal(a * lM + b, std, nsamp_Mdot)
            logMdot[i] = lMdot_med + offsets #collect all simulated points for that M
        return logMdot
    
    #if it's just one mass, do the same
    else:
        logMdot = np.zeros(nsamp_Mdot)
        lMdot_med = a * logM + b
        offsets = np.random.normal(0, std, nsamp_Mdot)
        logMdot = lMdot_med + offsets
        return logMdot

def calc_orbit(
  epochs, sma, ecc, inc, aop, pan, tau, mtot, tau_ref_epoch=58849, tolerance=1e-9, max_iter=100, use_c=True, use_gpu=False
):

    """
    Returns the separation and radial velocity of the body given array of
    orbital parameters (size n_orbs) at given epochs (array of size n_dates)

    Based on orbit solvers from James Graham and Rob De Rosa. Adapted by Jason Wang and Henry Ngo.

    Args:
        epochs (np.array): MJD times for which we want the positions of the planet
        sma (np.array): semi-major axis of orbit [au]
        ecc (np.array): eccentricity of the orbit [0,1]
        inc (np.array): inclination [radians]
        aop (np.array): argument of periastron [radians]
        pan (np.array): longitude of the ascending node [radians]
        tau (np.array): epoch of periastron passage in fraction of orbital period past MJD=0 [0,1]
        mtot (np.array): total mass of the two-body orbit (Mstar + Mplanet) [Solar masses]
        tau_ref_epoch (float, optional): reference date that tau is defined with respect to (i.e., tau=0)
        tolerance (float, optional): absolute tolerance of iterative computation. Defaults to 1e-9.
        max_iter (int, optional): maximum number of iterations before switching. Defaults to 100.
        use_c (bool, optional): Use the C solver if configured. Defaults to True
        use_gpu (bool, optional): Use the GPU solver if configured. Defaults to False

    Return:
        rad (np.array): array-like (n_dates x n_orbs) of offsets between bodies [AU]

    Written: Jason Wang, Henry Ngo, 2018
    Truncated by CP, 2022
    """
    # compute mean anomaly
    manom = orbitize.kepler.tau_to_manom(epochs, sma, mtot, tau, tau_ref_epoch)
    # compute eccentric anomalies (size: n_orbs x n_dates)
    eanom = orbitize.kepler._calc_ecc_anom(manom, ecc, tolerance=tolerance, max_iter=max_iter, use_c=use_c, use_gpu=use_gpu)

    # compute the true anomalies
    tanom = 2.*np.arctan(np.sqrt((1.0 + ecc)/(1.0 - ecc))*np.tan(0.5*eanom))
    # compute 3-D orbital radius of second body
    radius = sma * (1.0 - ecc * np.cos(eanom))
    
    # compute ra/dec offsets
    # math from James Graham. Lots of trig
    c2i2 = np.cos(0.5*inc)**2
    s2i2 = np.sin(0.5*inc)**2
    arg1 = tanom + aop + pan
    arg2 = tanom + aop - pan
    c1 = np.cos(arg1)
    c2 = np.cos(arg2)
    s1 = np.sin(arg1)
    s2 = np.sin(arg2)

    # updated sign convention for Green Eq. 19.4-19.7
    # CP: removed plx so this is just AU, not mas
    raoff_AU = radius * (c2i2*s1 - s2i2*s2)
    deoff_AU = radius * (c2i2*c1 + s2i2*c2)
    
    proj_rad = np.sqrt(raoff_AU**2 + deoff_AU**2)

    return proj_rad

def eccpdf(x):
    return 2.1-2.2*x
def eccprob(x):
    return scipy.integrate.quad(eccpdf,x,x+0.95/1001)[0]
eccvals = np.linspace(0,0.95,1000)
eccprobs = [eccprob(ev) for ev in eccvals]
eccprobs = [ep-(sum(eccprobs)-1)/len(eccprobs) for ep in eccprobs]

def sim_orb(mstar, mplanet, sma, n, returne = False, returni = False):
    """
    Simulate various planets around a star.
    Get the projected radius [AU] for each planet at a (usually randomized) point in its orbit.
    
    Args:
        mstar: stellar mass [M_sun]
        mplanet: planet mass [M_sun]
        sma: semimajor axis [au]
        n: number of orbits to simulate

        returne: boolean, return all simulated eccentricities (default: False)
        returni: boolean, return all simulated inclinations (default: False)
        
    Returns:
        proj_rads: array-like, projected radii between planet(s) and star [AU] for each simulated planet
        
        potentially in conjunction (array-like) with:
        eccs: array-like, simulated eccentricities (optional)
        incs: array-like, simulated inclinations (optional)
        
    """

    mtot = mstar + mplanet
    period_days = np.sqrt(sma**3/mtot) * 365.25 #kepler's 3rd for period
    # number of epochs at which to solve the orbit (we just need one)
    epochs = np.linspace(0, period_days, 1)

    ####################
     #simulate the orbital parameters based on known distributions
    ####################
    cosinc = np.random.rand(n)*2-1
    inc = np.arccos(cosinc) #rad
    aop = np.random.rand(n)*2*np.pi #rad
    ecc = np.random.choice(eccvals,p=eccprobs,size=n)
    #ecc=np.zeros(n)
    pan = np.zeros(n) #rad #don't need to simulate bc of azimuthal symmetry
    tau = np.random.rand(n)

    proj_rads = calc_orbit(epochs, sma, ecc, inc, aop, pan, tau, mtot)

    if returne:
        if returni: return np.array(proj_rads,ecc,inc)
        else: return np.array(proj_rads,ecc)
    elif returni: return np.array(proj_rads,inc)
    return proj_rads

#get r' band magnitudes, distances, ages, and masses
obs_data = pd.read_csv('obs_data_new.csv', index_col = 0)

#get contrast curves. Specific to GAPlanets, clearly
workdir = '/Users/cailinplunkett/Google Drive/Shared drives/Follette-Lab-AWS-2/contrasts'
obsdirs = glob.glob(workdir+'/*')

def get_stellar_ccs(star):
    ccs = {}
    for obsdir in obsdirs:
        if os.path.isdir(obsdir):
            if star in obsdir:
                date = obsdir.split('/')[-1][len(star)+1:]
                mostrec = 0 #initialize var to determine most recently modified file
                for file in glob.glob(obsdir+'/*contrasts.fits'):
                    lastmod = os.path.getmtime(file) #last modified time (higher = more recent)
                    if lastmod > mostrec:
                        mostrec = lastmod; ccf = file
                try:
                    with fits.open(ccf) as ccdata:
                        cc = np.array([ccdata[0].data[0][0] * 7.95, ccdata[0].data[0][2]])

                        fwhm_file = glob.glob(obsdir+'/*KLmodes-all.fits')[0]
                        fwhm = round(float(fits.getheader(fwhm_file)['FWHM']))
                        if star == 'HD100453':
                            if fwhm == 0: fwhm = 4

                        fnct = interpolate.interp1d(cc[0],np.log10(cc[1]),fill_value = 'extrapolate')
                        IWA_interpval = 10**fnct(fwhm * 7.95)

                        cc = np.insert(cc.T,0,[fwhm*7.95,IWA_interpval],axis=0).T
                        
                        #manually mask radii where HD 100453 b impacts the contrast curve
                        if star == 'HD100453':
                            if '2May18_short' in ccf: cc = np.delete(cc, np.arange(72,77),axis=1)
                            elif '2May18_long' in ccf: cc = np.delete(cc, np.arange(63,76),axis=1)
                            elif '3May18' in ccf: cc = np.delete(cc, np.arange(69,77),axis=1)
                            elif '17Feb17' in ccf: cc = np.delete(cc, np.arange(52,59),axis=1)
                    
                    #ccs.append(cc)
                    ccs[date] = cc
                except: continue
    if len(ccs) == 0:
        print(f'uh oh, problem loading in contrast curves for {star}')
        return
    return ccs

def get_Lacc(logMMdot, R = 2):
    '''
    get accretion luminosity for a planet.
    args:
        (args are scalars; fn will add astropy units)
        logMMdot: log product of mass and mass accretion [M_sun**2 / yr]
        R: planet radius [R_jup] (default: 2)
        
    returns:
        Lacc: accretion luminosity. unitless, Lacc/Lsun.
    '''
    MMdot = 10**logMMdot * (u.M_sun**2) * (u.year)**(-1)
    R = R * u.R_jup #add units
    #typical function for Lacc. Assumes magnetospheric truncation radius of 5R and all energy to radiation
    #probably less accurate for non magnetospheric accretion, but what would apply is less clear.
    return (MMdot * c.G / (1.25 * R * u.Lsun)).decompose().value

def get_logLHa(Lacc, reln = 'alcala'):
    if reln == 'alcala' or reln == 'empirical':
        return (np.log10(Lacc) - 1.74) / 1.13
    elif reln == 'aoyama' or reln == 'theoretical':
        return (np.log10(Lacc) - 1.61) / 0.95
    else:
        print('uh oh! unknown scaling relation :(')
        return

#telescope parameters. Specific to GAPlanetS
z = 0.0063 * u.um
dlambda = 1.733e-5 * u.erg / u.cm**2 / u.s / u.um

def get_Ha_contrast(star, log_LHa, d, mag_star):
    
    LHa = 10**log_LHa * u.Lsun
    FHa = (LHa / (4 * np.pi * (d * u.pc)**2 * z * dlambda)).decompose().value
    logFHa = np.log10(FHa)
    extinction_correction = (mag_star - obs_data['A_r'][star])/2.5
    contrasts = {}
    scale_factors = ast.literal_eval(obs_data['S'][star])
    for night in scale_factors.keys():
        s = scale_factors[night]
        logC = logFHa + extinction_correction - np.log10(s)
        contrasts[night] = 10**logC
    return contrasts

def detectable(star, proj_sep, C, contrastcurves, ps_bins, num_psbins):
    """
    determines if a simulated planet is detectable for a given set of contrast curves
    args:
        proj_sep: (array-like) projected separation(s) of companion [mas]
        C: (array-like) Ha contrast(s) of companion to star
        contrastcurves: contrast curve(s) of target star. array-like with (projected separation, contrast) for each curve.
            a companion is detectable if it is above at least one contrast curve.
    returns:
        is_detectable: boolean (array-like), True if (projected separation, contrast) is above contrast curve, False otherwise.
    """
    #stores whether each simulated object is detectable
    is_detectable = np.empty(len(proj_sep), dtype='bool')
    #stores how many detectable objects there are in each separation bin
    det_sep_counts = np.zeros(num_psbins)
    
    #iterate over simulated objects:
    for i, ps in enumerate(proj_sep):
        for night in contrastcurves.keys():
            cc = contrastcurves[night]
            #first, check if projected separation is outside our view: too far or too close
            if ps > cc[0][-1] or ps < cc[0][0]:
                is_detectable[i] = False
            else:
                #interpolate to get contrast curve value at planet's projected separation
                detection_limit = np.interp(ps, cc[0], cc[1])
                is_detectable[i] = C[night][i] > detection_limit
                #if it's detectable at least once, we're good!
                if is_detectable[i]:
                    det_sep_counts[np.digitize(ps, ps_bins)-1] +=1 
                    break
               
    #return the overall detectable fraction and the number detectable in each bin
    return sum(is_detectable) / len(is_detectable), det_sep_counts


def main(star, formationparadigm = 'empirical', accretionparadigm = 'empirical',
                     logsma_min = 0, logsma_max = 2.7, num_sma = 10,
                     yax_min = -3, yax_max = -0.5, num_yax = 10,
                     num_sim = 100, num_psbins = 30, outdir='', returngrid = True):
    """
    args:
        star: what star! choose a nice name from the gapplanets survey.
            or, put an array of stars to make a map for each.
            or, put 'all' to make a map for all fourteen stars.
        
        formationparadigm: how to relate M and Mdot. Options are 'empirical' (default), which uses CASPAR
            to simulate Mdot from M; 'theoretical', which uses Stamatellos + Herczeg disk fragmentation model;
            or 'agnostic', which simulates the product MMdot.
        accretionparadigm: how to relate Lacc to LHa. Options are 'empirical' (default), which uses the Alcala
            T-Tauri scaling relation; or 'theoretical', which uses the Aoyama planetary relation.
        
        logsma_min: minimum semimajor axis to simulate (default: 0 log AU)
        logsma_max: maximum semimajor axis to simulate (default: 3 log AU)
        num_sma: number of semimajor axes to simulate (log-spaced) (default: 10)
        
        yax_min: minimum y-axis value to simulate. [log M_sun] for 'empirical' or 'theoretical' formationparadigm,
            [log M_sun^2 / yr] for 'agnostic' formationparadigm. (default: -4)
        yax_max: maximum y-axis value, as above. (default:-4)
        num_yax: number of log-spaced y-axis values to simulate (default: 10)
                
        num_sim: number of planets to simulate for each (x,y) pair (default: 100).
            for each simulated object, it generates an orbit. If formationparadigm is not agnostic, it also assigns
            an accretion rate to each object.
        
        num_psbins: how many projected separation bins (log-spaced) to sort detectable planets into
            
        outdir: where 2 put the grids
        
        returngrid: return the grid, or don't. 
        
            
    returns:
        percent_detectable_grid: percent of planets detectable for each (x,y) pair.
            in a 2d array with shape (num_yax, num_sma).
            
        also writes each grid to a csv in outdir
    """
    
    #make lists of x and y parameter values
    logsma_vals = np.linspace(logsma_min, logsma_max, num_sma)
    yax_vals = np.linspace(yax_min, yax_max, num_yax) 
    
    #make all same dtype
    if type(star)==str:
        if star in obs_data.index: star = np.array([star])
        elif star == 'all': star = obs_data.index
    print(f'running star(s) {star}')
    
    #get contrast curves
    print('loading stellar ccs')
    contrastcurves = {st: get_stellar_ccs(st) for st in star}
    
    #define projected separation bins: we care about 10 to 2200 mas
    ps_bins = np.logspace(1,np.log10(2200),num_psbins+1)
    
    #initialize percent detectable
    print('initializing arrays')
    percent_detectable = np.zeros((len(star), num_yax, num_sma))
    num_detectable_ps = np.zeros((len(star), num_yax, num_sma, num_psbins))
    num_sorted_ps = np.zeros((len(star), num_yax, num_sma, num_psbins))
    
    for i, logsma in enumerate(logsma_vals):
        
        #Projected separations being independent of the star and planet masses is a good approximation
        proj_seps_AU = sim_orb(mstar = 1, mplanet = 0.01, sma = 10**logsma, n = num_sim)
        
        for j, yval in enumerate(yax_vals):
            
            if formationparadigm != 'agnostic':
                logMdots = get_companions(yval, nsamp_Mdot = num_sim, formpar = formationparadigm)

            for k, st in enumerate(star):
                rmag = obs_data['r\' magnitude'][st]
                d = obs_data['distance [pc]'][st]
                
                #compute projected angular separations and count how many fall into each bin
                #note: it's OK if planets are below 10 mas. they just won't be counted here.
                proj_seps = 1000 * proj_seps_AU / d
                ps_bin_counts = np.histogram(proj_seps,ps_bins)[0]
                
                #get accretion luminosities
                if formationparadigm == 'agnostic': Laccs = np.ones(num_sim) * get_Lacc(yval) #yval is log MMdot here
                else: Laccs = get_Lacc(yval + logMdots) #add logM to logMdots

                #get Ha luminosities
                logLHas = get_logLHa(Laccs, accretionparadigm)

                #contrasts
                Cs = get_Ha_contrast(st,logLHas, d, rmag)

                #detectability
                contrastcurves_star = contrastcurves[st]
                #get overall percent detectable at the given (a, M) value, and the number of detectable objects in each projected separation bin
                percdet, ps_det_counts = detectable(st, proj_seps, Cs, contrastcurves_star, ps_bins = ps_bins, num_psbins = num_psbins)
                
                percent_detectable[k, j, i] = percdet                
                num_sorted_ps[k, j, i, :] = ps_bin_counts
                num_detectable_ps[k, j, i, :] = ps_det_counts
    
    num_sort_ps_reshaped = num_sorted_ps.reshape((len(star),num_yax,num_sma*num_psbins))
    num_det_ps_reshaped = num_detectable_ps.reshape((len(star),num_yax,num_sma*num_psbins))

    for s, percdet_grid, numsort_projsep_grid, numdet_projsep_grid in zip(star, percent_detectable, num_sort_ps_reshaped, num_det_ps_reshaped):
        fn = f'{outdir}{s}-formpar_{formationparadigm}-accrpar_{accretionparadigm}-logsma_{logsma_min}_{logsma_max}-logy_{abs(yax_min)}_{abs(yax_max)}-gridsize_{num_sma}-numsim_{num_sim}.csv'
        fn_ps_ns = f'{outdir}{s}-formpar_{formationparadigm}-accrpar_{accretionparadigm}-logsma_{logsma_min}_{logsma_max}-logy_{abs(yax_min)}_{abs(yax_max)}-gridsize_{num_sma}-numsim_{num_sim}-numsort.csv'
        fn_ps_nd = f'{outdir}{s}-formpar_{formationparadigm}-accrpar_{accretionparadigm}-logsma_{logsma_min}_{logsma_max}-logy_{abs(yax_min)}_{abs(yax_max)}-gridsize_{num_sma}-numsim_{num_sim}-numdet.csv'
        np.savetxt(fn,percdet_grid,delimiter=',')
        np.savetxt(fn_ps_ns,numsort_projsep_grid,delimiter=',')
        np.savetxt(fn_ps_nd,numdet_projsep_grid,delimiter=',')

if __name__ == '__main__':

    # execute only if run as a script
    import sys
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    parser.add_argument("--star", type=str, default='all',help="what star to run. all or one, for now")
    parser.add_argument("--formpar", default='empirical', help="what M to Mdot relation")
    parser.add_argument("--accpar", default='empirical', help="what LHa to Lacc relation")
    parser.add_argument("--logsma_min",default=0,help="log of min semimajor axis to simulate")
    parser.add_argument("--logsma_max",default=2.7,help="log of max semimajor axis to simulate")
    parser.add_argument("--num_sma", default=60,help="number of semimajor axes to simulate (log-spaced)")
    parser.add_argument("--yax_min", default=-3,help="log of min y-axis value to simulate")
    parser.add_argument("--yax_max", default=-0.5,help="log of max y-axis value to simulate")
    parser.add_argument("--num_yax", default=60,help="number of y axis values to simulate (log-spaced)")
    parser.add_argument("--num_psbins", default=60,help=" how many projected separation bins (log-spaced) to sort detectable planets into")
    parser.add_argument("--num_sim", default=10000,help="number of planets to simulate for each (x,y) pair")
    parser.add_argument("--outdir",default='.',help="where to put the output")
    
    ops= parser.parse_args()
    outdir=ops.outdir
    if not os.path.isdir(outdir):
        os.makedirs(outdir) 
    
    main(ops.star,formationparadigm = ops.formpar, accretionparadigm = ops.accpar,
                        logsma_min = float(ops.logsma_min), logsma_max = float(ops.logsma_max), num_sma = int(ops.num_sma),
                        yax_min = float(ops.yax_min), yax_max = float(ops.yax_max), num_yax = int(ops.num_yax),
                        num_sim = int(ops.num_sim), num_psbins = int(ops.num_psbins), outdir = ops.outdir)