import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as mpl
import astropy.io.fits as pyfits
import subprocess as sp
from scipy.integrate import simps
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import scipy.ndimage as snd
import matplotlib.gridspec as gridspec
import matplotlib.widgets as mpw
from matplotlib.patches import Rectangle
import matplotlib.colors as mpc

import sys
import os
import argparse

mpl.rcParams['keymap.fullscreen']=''
mpl.rcParams['keymap.home']=''
mpl.rcParams['keymap.back']=''
mpl.rcParams['keymap.forward']=''
mpl.rcParams['keymap.pan']=''
mpl.rcParams['keymap.zoom']=''
mpl.rcParams['keymap.save']=''
mpl.rcParams['keymap.grid']=''
mpl.rcParams['keymap.yscale']=''
mpl.rcParams['keymap.xscale']=''
mpl.rcParams['keymap.all_axes']=''
mpl.rcParams['xtick.labelsize']=12
mpl.rcParams['ytick.labelsize']=12

W,L,S,E = np.loadtxt('EMISSION_LINES.txt',unpack=True,usecols=[0,1,2,3],dtype='f4,a20,f4,f4')

""" COSMOLOGY MODULE"""

#### PLACNK COSMOLOGY
Obar=0.049                  #Baryon Density Parameter
Omat=0.3175                 #Matter Density Parameter
Ok=0.0                      #Curvature Density Parameter
Orad=0.0                    #Radiation Density Parameter
Ow=0.6825                   #Dark Energy Density Parameter
w=-1.0                      #DE equation of state parameter p/rho=w
H0=67.11                    #km/s/Mpc
Msun=1.989e30               #kg
Mpc=3.0857e19               #km
c=2.9979e5                  #km/s

def Hubble(z,pars=None):
    "Returns the value for the standard Hubble parameter at a redshift z"
    P={'h':H0/100,'r':Orad,'m':Omat,'k':Ok,'l':Ow,'w':w}
    if not (pars==None):
        for p in pars:
            P[p] = pars[p]
    return 100*P['h']*np.sqrt(P['r']*(1+z)**4.+P['m']*(1+z)**3.+P['k']*(1+z)**2.+P['l']*(1+z)**(3*(1.+P['w'])))

def comov_rad(z,pars=None,npoints=10000):
    """Returns the comoving radial distance corresponding to the redshift z in Mpc
    If nedeed, multiply by h to get result in units of (h**-1 Mpc)"""
    if z==0:
        z=1e-4 
    radius=[]
    z_points=np.linspace(1e-5,z,npoints)
    H=Hubble(z_points,pars=pars)
    invH=1./H
    radius=simps(invH,z_points)
    return c*radius
    
def angular_distance(z,pars=None):
    "Computes the angular diameter distance (Mpc) in a standard LCDM cosmology"
    return comov_rad(z,pars=pars)/(1+z)

def luminosity_distance(z,pars=None):
    "Computes the luminosity distance (Mpc) in a standard LCDM cosmology"
    return comov_rad(z,pars=pars)*(1+z)

"""END COSMOLOGY MODULE"""    


def read_datacube(name):
    try:        
        filename = sp.check_output('ls %s.V500.rscube.fits'%name,shell=True,stderr=sp.PIPE).split()[0]    
        hdu = pyfits.open(filename)
        data = hdu[0].data
        hdr = hdu[0].header
        wave = hdr['CRVAL3']+np.arange(data.shape[0])*hdr['CDELT3']
        return hdu,wave,data,hdr
    except sp.CalledProcessError:
        return None,None,None,None

def collapse_spectra_all(data,badpix):
    D = data.copy()    
    D[badpix==1]=0
    return np.sum(np.sum(D,axis=1),axis=1)

def gaussian(x,a,c,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))+c
    
def mask_spectra(redshift,line_in, spec_in,wave_line,slim=10,kmask=4): #para mascarar tudo menos a linha que se quer ajustar
    masked_line = line_in.copy().astype(np.float64)
    masked_spec = spec_in.copy().astype(np.float64)
    for (line_peak,line_strength) in zip(W,S):
        if line_peak != wave_line and line_strength>=slim:
            try:
                k = np.where(masked_line>line_peak*(1+redshift))[0][0]
                masked_spec[k-kmask:k+kmask] = np.nan
                masked_line[k-kmask:k+kmask] = np.nan
            except IndexError:
                continue

    masked_wave = (np.isnan(masked_line) == False)
    
#    fig,ax=mpl.subplots()
#    ax.plot(masked_line[masked_wave], masked_spec[masked_wave])
    
    return masked_line[masked_wave], masked_spec[masked_wave]
    
def draw_lines(eixo,redshift,wavelength,spec,strength):
    for l in range(len(W)):
        if W[l]*(1+redshift) < min(wavelength) or W[l]*(1+redshift) > max(wavelength):
            continue
        if S[l] > strength:
            if E[l] == 0.0:
                ls = '--'
            elif E[l] == 0.5:
                ls = ':'
            else:
                ls = '-.'
            eixo.vlines(W[l]*(1+redshift),1.1*min(spec),1.1*max(spec),linestyle=ls,linewidth=np.log10(S[l]+2),color='Crimson')
            eixo.text(W[l]*(1+redshift)+5,1.0*max(spec),L[l],ha='left',va='top',fontsize=8,rotation=270,color='Crimson')
        else:
            continue
    return None    

#def get_line_flux(wave,spectra,wave_line,wave_start,wave_end,contpoints=3,linepoints=5):    
#    k1 = np.where(wave>wave_start)[0][0]
#    k2 = np.where(wave>wave_end)[0][0]
#    kline = np.where(wave>wave_line)[0][0]
#    
#    def contline(x,m,b):
#        return m*x+b
#    
#    mask_wave,mask_spec = mask_spectra(wave[k1:k2],spectra[k1:k2],wave_line)
#    
#    cpars,cerr = curve_fit(contline,np.append(mask_wave[:contpoints],mask_wave[-contpoints:]),np.append(mask_spec[:contpoints],mask_spec[-contpoints:]))
#    continuum = contline(wave,*cpars)
#
#    s_line = spectra[kline-linepoints+1:kline+linepoints+1]
#    c_line = continuum[kline-linepoints+1:kline+linepoints+1]
#    w_line = wave[kline-linepoints+1:kline+linepoints+1]
#    flux = simps(s_line-c_line,w_line)
#
#
#    if VISUALCHECK:
#        mpl.figure(figsize=(22,8))
#        gs = gridspec.GridSpec(1, 2, width_ratios=[2,1])
#        ax=[mpl.subplot(gs[0]),mpl.subplot(gs[1])]
#        mpl.subplots_adjust(wspace=0.0)
#        ax[0].plot(cube_wave,spec1D,'b-')
#        ax[0].set_xlim(cube_wave[0],cube_wave[-1])
#        draw_lines(ax[0],spec1D,15)
#        ax[0].set_ylim(1.1*min(spec1D),1.1*max(spec1D))
#        
#        ax[1].plot(wave[k1:k2],spectra[k1:k2],'b-')
#        ax[1].plot(mask_wave,mask_spec,'k-')
#        ax[1].plot(wave[k1:k2],continuum[k1:k2],'r--')
#        ax[1].fill_between(w_line,c_line,s_line,color='green',alpha=0.2)
#        ax[1].tick_params(axis='y', which='both', labelleft='off', labelright='on')
#        ax[1].set_xlim(mask_wave[0],mask_wave[-1])
#    
#    try:
#        gauss_pars,gauss_errs = curve_fit(gaussian,mask_wave,mask_spec,p0=[1,10,wave_line,1])
#        flux_gauss = simps(gaussian(wave[k1:k2],*gauss_pars)-gauss_pars[1],wave[k1:k2])
#        fwhm_gauss = 2*np.sqrt(2*np.log(2))*abs(gauss_pars[3])
#        if VISUALCHECK:
#            ax[1].plot(wave[k1:k2],gaussian(wave[k1:k2],*gauss_pars),'b--')
#    except RuntimeError:
#        flux_gauss=-99
#        fwhm_gauss=-99
#
#    if VISUALCHECK:
#        mpl.show(block=False)
#
#        ans = raw_input('Is there a good fit to the line? [y/n]\t')
#        if ans == 'Y' or ans == 'y':
#            pass
#        else:
#            flux=-99
#            flux_gauss=-99
#            fwhm_gauss=-99
#    
#        mpl.close('all')
#
#    return flux,flux_gauss,fwhm_gauss

def make_line_map(wave,data,wave_line,linepoints=5):
    kline = np.where(wave>wave_line)[0][0]
    data_line = data[kline-linepoints:kline+linepoints,:,:]
    return np.sum(data_line,axis=0)

def make_filter_image(wave,data,filter_center,filter_width):
    k1 = np.where(wave>filter_center-filter_width/2.)[0][0]
    k2 = np.where(wave>filter_center+filter_width/2.)[0][0]    
    data_line = data[k1:k2,:,:]
    return np.sum(data_line,axis=0)

def vacuum_to_air(wave_vac):
    """equation from Donald Morton (2000, ApJ. Suppl., 130, 403),wavelength in Angstrom."""
    s = 1e4/wave_vac
    n = 1.0 + 0.0000834254 + 0.02406147 / (130 - s*s) + 0.00015998 / (38.9 - s*s)
    wave_air = wave_vac/n
    return wave_air
    
class ObjectInfo:
    def __init__(self,ID,linenames,linemults):
        self.ID=ID
        self.nlines = len(linenames)
        self.linemults=linemults

        self.line_center= np.array([np.array([0.]*g) for g in linemults])-999.
        self.line_center_err= np.array([np.array([0.]*g) for g in linemults])-999.

        self.flux = np.zeros(self.nlines)-999.
        self.flux_gauss=np.zeros(self.nlines)-999.
        self.flux_gauss_err  = np.zeros(self.nlines)-999.

        self.fwhm = np.array([np.array([0.]*g) for g in self.linemults])-999.
        self.fwhm_err = np.array([np.array([0.]*g) for g in self.linemults])-999.
        self.amplitude = np.array([np.array([0.]*g) for g in self.linemults])-999.        
        self.amplitude_err = np.array([np.array([0.]*g) for g in self.linemults])-999.
        self.integral = np.array([np.array([0.]*g) for g in self.linemults])-999.        
        self.integral_err = np.array([np.array([0.]*g) for g in self.linemults])-999.        
        
        self.luminosity = np.zeros(self.nlines)-999.
        self.luminosity_gauss=np.zeros(self.nlines)-999.
        self.luminosity_gauss_err=np.zeros(self.nlines)-999.
        
        self.eqw=np.zeros(self.nlines)-999.
        self.eqw_gauss=np.zeros(self.nlines)-999.
        self.eqw_gauss_err=np.zeros(self.nlines)-999.

        self.signal_to_noise = np.zeros(self.nlines)-999.
        return None
    
    def __str__(self):
        s = '%s'%self.ID
        for n in range(self.nlines):
            s+= "\t%s"%('[%s]'%(','.join(['%.4f'%(value) for value in self.line_center[n]])))
            s+= "\t%s"%('[%s]'%(','.join(['%.4f'%(value) for value in self.line_center_err[n]])))
            s+= "\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e"%(self.flux[n],self.luminosity[n],self.flux_gauss[n],self.flux_gauss_err[n],self.luminosity_gauss[n],self.luminosity_gauss_err[n])
            s+= "\t%s"%('[%s]'%(','.join(['%.4f'%(value) for value in self.fwhm[n]])))
            s+= "\t%s"%('[%s]'%(','.join(['%.4f'%(value) for value in self.fwhm_err[n]])))
            s+= "\t%s"%('[%s]'%(','.join(['%.4f'%(value) for value in self.amplitude[n]])))
            s+= "\t%s"%('[%s]'%(','.join(['%.4f'%(value) for value in self.amplitude_err[n]])))
            s+= "\t%s"%('[%s]'%(','.join(['%.4f'%(value) for value in self.integral[n]])))
            s+= "\t%s"%('[%s]'%(','.join(['%.4f'%(value) for value in self.integral_err[n]])))
            s+= "\t%.4f\t%.4f\t%.4f\t%.4f"%(self.eqw[n],self.eqw_gauss[n],self.eqw_gauss_err[n],self.signal_to_noise[n])
        return s 
    
    def nulify(self):
        self.line_center= np.array([np.array([0.]*g) for g in self.linemults])-99.
        self.line_center_err= np.array([np.array([0.]*g) for g in linemults])-99.

        self.flux = np.zeros(self.nlines)-99.
        self.flux_gauss=np.zeros(self.nlines)-99.

        self.fwhm = np.array([np.array([0.]*g) for g in self.linemults])-99.
        self.fwhm_err = np.array([np.array([0.]*g) for g in self.linemults])-99.
        self.amplitude = np.array([np.array([0.]*g) for g in self.linemults])-99.        
        self.amplitude_err = np.array([np.array([0.]*g) for g in self.linemults])-99.
        self.integral = np.array([np.array([0.]*g) for g in self.linemults])-99.        
        self.integral_err = np.array([np.array([0.]*g) for g in self.linemults])-99.        
        
        self.luminosity = np.zeros(self.nlines)-99.
        self.luminosity_gauss=np.zeros(self.nlines)-99.
        self.luminosity_gauss_err=np.zeros(self.nlines)-99.
        
        self.eqw=np.zeros(self.nlines)-99.
        self.eqw_gauss=np.zeros(self.nlines)-99.
        self.eqw_gauss_err=np.zeros(self.nlines)-99.

        self.noise = np.zeros(self.nlines)-99.

        return None
        
class SimpleWidget:

    def __init__(self,table,index,lines,linenames,linemults):
        self.normflux=1e-18
        self.pixscale = 0.03
        self.spec_disp = 0.6 # AA/pixel
        self.spec_resol = 20*self.spec_disp # pixel
        self.window = 150  ## AA size of zoom in on line
        self.sky_threshold = 0.2
        
        self.table = table
        self.lines = lines
        self.linenames = linenames
        self.linemults = linemults

        self.get_spectra_names(self.table)

        if os.path.isfile(TABLE_NAME):
            self.load_table(TABLE_NAME)
        else:
            self.ObjectTable = dict()
        
        self.load_spectra(index)
        
        self.fig = mpl.figure(figsize=(17,10))
        
        self.index=index
        self.index_line = 0
        
        self.line_shift=0
        self.step_style=False
        self.COLORMAP='viridis'
        self.SPECCOLOR='RoyalBlue'
        self.CONTCOLOR='ForestGreen'
        self.CONTPOINTCOLOR='DarkGreen'
        self.GAUSSCOLOR='Crimson'
        self.FILLCOLOR='BlueViolet'        
        self.NOISECOLOR='Gainsboro'
        self.SLITCOLOR='LightBlue'
        self.BUTTONCOLOR='LightSteelBlue'
        self.SLIDERCOLOR='LightSteelBlue'
        self.WAVEMARKCOLOR='red'

        self.set_ohlines()
        self.interactive_measure_single_line(self.lines[self.index_line],window=self.window)
        mpl.show()

    def set_ohlines(self):
        colors = [(1,1,1,i) for i in np.linspace(1,0,10)]
        self.OHcmap = mpc.LinearSegmentedColormap.from_list('OHcmap', colors, N=10)
        wave,ohline = np.loadtxt('OH_Lines.txt',unpack=True)
##        wave = vacuum_to_air(wave) ## convert from vacuum to air wavelengths
        self.OHwave = np.arange(wave[0],wave[-1],self.spec_disp)
        self.delta_OHwave = (self.OHwave[1]-self.OHwave[0])/2
        oh_interpolator = interp1d(wave,ohline,bounds_error=False,fill_value=0)
        new_oh = oh_interpolator(self.OHwave)
        self.OHintensity = np.zeros([1,len(self.OHwave)])
        self.OHintensity[0,:]=new_oh
        self.OHintensity_masked = np.ma.masked_where(self.OHintensity<=self.sky_threshold,self.OHintensity,copy=False) 
        return None
           
    def get_spectra_names(self,table):
        self.specnames1D = []
        self.specnames2D = []
        self.stampnames = []
        for i in range(len(self.table)):
            self.specnames1D.append('pointing%i/stack-Q%i/spec1d%03i_calib'%(self.table['Pointing'][i],self.table['Quadrant'][i],self.table['ObjID'][i]))
            self.specnames2D.append('pointing%i/stack-Q%i/spec2d.P%i.%03i.Q%i.fits'%(self.table['Pointing'][i],self.table['Quadrant'][i],self.table['Pointing'][i],self.table['ObjID'][i],self.table['Quadrant'][i]))
            self.stampnames.append('stamps/P%i_Q%i_S%03i.fits'%(self.table['Pointing'][i],self.table['Quadrant'][i],self.table['ObjID'][i]))
        return None
    
    def load_spectra(self,index):
#        rad_tel = 400 #centimeters
#        h=6.626e-27 #ergs.s
#        c=2.9979e10 #cm/s

        try:
            wavelength, spec1D, spec1d_err = np.loadtxt(self.specnames1D[index],unpack=True)
        except IOError:
            print("Skipping due to lack of calibration data")
            self.save_to_table()
            self.fig.clf()
            self.index+=1
            self.load_spectra(self.index)
            self.interactive_measure_single_line(self.lines[self.index_line],window=self.window)
            
        self.wavelength = wavelength[spec1D!=0]
        self.spec1D = spec1D[spec1D!=0]/self.normflux#/(np.pi*rad_tel**2)*(h*c/(self.wavelength*1e-8))/5.1e-8
        self.spec1d_err = spec1d_err[spec1D!=0]

        spec2D = pyfits.getdata(self.specnames2D[index])
        self.spec2D = spec2D[:,spec1D!=0]

        self.stampI = pyfits.getdata(self.stampnames[index])
        return None
#'Pointing', 'Quadrant', 'ObjID'

    def get_mouse_position(self,event):
        if event.inaxes == self.ax:
            pass
        else:
            pass
        return None
        
    def gaussian(self,x,a,x0,sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
    
    def gaussian2(self,x,a1,x1,s1,a2,x2,s2):
        return self.gaussian(x,a1,x1,s1)+self.gaussian(x,a2,x2,s2)

    def gaussian3(self,x,a1,x1,s1,a2,x2,s2,a3,x3,s3):
        return self.gaussian(x,a1,x1,s1)+self.gaussian(x,a2,x2,s2)+self.gaussian(x,a3,x3,s3)

    def get_button_press(self,event):
        if event.inaxes == self.ax:
            self.Xpress,self.Ypress=event.xdata,event.ydata
            if self.mode == 'continuum':

                for line in [self.cont_intblue,self.cont_intred]:
                    xl,xu = line.get_xdata()
                    if xl<self.Xpress<xu:
                        self.active_contline=line
                        break
                    else:
                        self.active_contline,= self.ax.plot([0,0],[-99,-99])

        return None
    
    def get_button_release(self,event):
        if event.inaxes == self.ax:
            self.Xrelease,self.Yrelease=event.xdata,event.ydata
            
            if self.mode == 'mask':
                if self.Xpress<self.Xrelease:
                    kstart = np.where(self.wave>self.Xpress)[0][0]
                    kend = np.where(self.wave>self.Xrelease)[0][0]
                else:
                    kstart = np.where(self.wave>self.Xrelease)[0][0]
                    kend = np.where(self.wave>self.Xpress)[0][0]
                    
                slope = (self.Yrelease-self.Ypress)/(self.Xrelease-self.Xpress)      
                yintercept = self.Yrelease-slope*self.Xrelease
                self.spec[kstart:kend] = self.wave[kstart:kend] * slope + yintercept
                self.specline.set_data([self.wave,self.spec])

            if self.mode == 'continuum':
                newx = self.Xrelease
                if self.active_contline == self.cont_intblue:
                    self.limbl = newx
                elif self.active_contline == self.cont_intred:
                    self.limrd = newx
                else:
                    pass
                    
                self.active_contline.set_xdata([newx-self.spec_resol,newx+self.spec_resol])

                self.continuum_line = self.estimate_continuum(self.wave,self.spec)
                self.contline.set_data([self.wave,self.continuum_line])
                self.fig.canvas.draw_idle()
        
                  
#                slope = (self.Yrelease-self.Ypress)/(self.Xrelease-self.Xpress)      
#                yintercept = self.Yrelease-slope*self.Xrelease
#                self.continuum_line = self.wave * slope + yintercept
#                self.contline.set_data([self.wave,self.continuum_line])


            if self.mode == 'line':
                self.line_shift = self.Xrelease-self.wave_line
                self.wave_line = self.Xrelease
                self.wave_mark.set_segments( [np.array([[self.wave_line,0.9*np.amin(self.spec)],[self.wave_line,1.1*np.amax(self.spec)]])] )
                if 'OII' in self.linenames[self.index_line]:
                    self.wave_mark2.set_segments([np.array([[self.Xrelease+deltaOII*(1+self.redshift),0.9*np.amin(self.spec1D[self.kstart:self.kend])],\
                                                            [self.Xrelease+deltaOII*(1+self.redshift),1.1*np.amax(self.spec1D[self.kstart:self.kend])]])])

            if self.mode == 'integrate':
                if self.Xpress<self.Xrelease:
                    kstart = np.where(self.wave>self.Xpress)[0][0]
                    kend = np.where(self.wave>self.Xrelease)[0][0]
                else:
                    kstart = np.where(self.wave>self.Xrelease)[0][0]
                    kend = np.where(self.wave>self.Xpress)[0][0]
                
                self.ObjectTable[self.name].flux[self.index_line]=simps(self.spec[kstart:kend]-self.continuum_line[kstart:kend],self.wave[kstart:kend])*self.normflux
                self.ObjectTable[self.name].eqw[self.index_line]=simps(1-self.spec[kstart:kend]/self.continuum_line[kstart:kend],self.wave[kstart:kend])
                self.ObjectTable[self.name].luminosity[self.index_line]=self.compute_lum_line(self.ObjectTable[self.name].flux[self.index_line])
                self.update_text()

                if self.step_style is True:
                    self.background = self.ax.fill_between(self.wave[kstart:kend],self.continuum_line[kstart:kend],self.spec[kstart:kend],color='dodgerblue',alpha=0.75,zorder=-1,step='mid')
                else:
                    self.background = self.ax.fill_between(self.wave[kstart:kend],self.continuum_line[kstart:kend],self.spec[kstart:kend],color='dodgerblue',alpha=0.75,zorder=-1,step=None)

            self.fig.canvas.draw_idle()
                
##    def __str__(self):
##        return "%s\t%.2f\t%.4e\t%.4e\t%.4e\t%.4e\t%4f"%(self.name,self.line_center,self.flux,self.luminosity,self.flux_gauss,self.luminosity_gauss,self.fwhm)

    def reset_data(self,event):
        self.specline.set_data([self.wavelength[self.kstart:self.kend],self.spec1D[self.kstart:self.kend]])
        self.slider_smooth.reset()
#        self.slide_scale.reset()
        self.fig.canvas.draw_idle()
    
    def fit_gaussian_to_data(self):
        
        if self.glevel==1:
            self.fitting_func=self.gaussian
            p0=[np.amax(self.spec-self.continuum_line),self.wave_line+self.line_shift,1]
        elif self.glevel==2:
            self.fitting_func=self.gaussian2
            p0=[np.amax(self.spec-self.continuum_line),self.wave_line+self.line_shift,1]+\
               [np.amax(self.spec-self.continuum_line),self.wave_line+self.line_shift+deltaOII*(1+self.redshift),1]
        elif self.glevel==3:
            self.fitting_func=self.gaussian3
            p0=[np.amax(self.spec-self.continuum_line),self.wave_line+self.line_shift,1]+\
               [np.amax(self.spec-self.continuum_line),self.wave_line+self.line_shift+deltaOII*(1+self.redshift),1]+\
               [np.amax(self.spec-self.continuum_line),self.wave_line+self.line_shift+2*deltaOII*(1+self.redshift),1]
        else:
            raise NotImplemented("%i number of gaussians not implemented"%self.glevel)
        
            
        self.gauss_pars,self.gauss_errs = curve_fit(self.fitting_func,self.wave,self.spec-self.continuum_line,p0=p0)
        func_cont = interp1d(self.wave,self.continuum_line,kind='linear')
        self.gaussline.set_data([self.wave_gauss,self.fitting_func(self.wave_gauss,*self.gauss_pars)+func_cont(self.wave_gauss)])

        self.integrate_gaussian()
        self.fig.canvas.draw_idle()
    
    def integrate_gaussian(self):
        
        FWHM=np.array([np.abs(self.gauss_pars[2+3*k]*2*np.sqrt(2*np.log(2))) for k in range(self.glevel)])
        FWHM_ERR=np.array([np.abs(self.gauss_errs[2+3*k,2+3*k]*2*np.sqrt(2*np.log(2))) for k in range(self.glevel)])
        self.ObjectTable[self.name].fwhm[self.index_line] = FWHM
        self.ObjectTable[self.name].fwhm_err[self.index_line] = FWHM_ERR
        AMPLITUDE =  np.array([np.abs(self.gauss_pars[0+3*k]) for k in range(self.glevel)])
        AMPLITUDE_ERR =  np.array([np.sqrt(self.gauss_errs[0+3*k,0+3*k]) for k in range(self.glevel)])
        self.ObjectTable[self.name].amplitude[self.index_line] = AMPLITUDE
        self.ObjectTable[self.name].amplitude_err[self.index_line] = AMPLITUDE_ERR
        INTEGRAL = np.array([AMPLITUDE[k]/(2*FWHM[k])*np.sqrt(np.pi/np.log(2)) for k in range(self.glevel)])
        INTEGRAL_ERR = np.array([AMPLITUDE_ERR[k]/(2*FWHM[k])*np.sqrt(np.pi/np.log(2))+AMPLITUDE[k]/(2*FWHM[k]**2)*np.sqrt(np.pi/np.log(2))*FWHM_ERR[k] for k in range(self.glevel)])
        self.ObjectTable[self.name].integral[self.index_line] = INTEGRAL
        self.ObjectTable[self.name].integral_err[self.index_line] = INTEGRAL_ERR

        self.ObjectTable[self.name].flux_gauss[self.index_line] = simps(self.fitting_func(self.wave,*self.gauss_pars),self.wave)*self.normflux
        self.ObjectTable[self.name].flux_gauss_err[self.index_line] = np.sqrt(np.sum(INTEGRAL_ERR*INTEGRAL_ERR))#simps(self.fitting_func(self.wave,*self.gauss_pars),self.wave)*self.normflux
        self.ObjectTable[self.name].line_center[self.index_line] = [self.gauss_pars[1+3*k] for k in range(self.glevel)]
        self.ObjectTable[self.name].line_center_err[self.index_line] = [np.sqrt(self.gauss_errs[1+3*k,1+3*k]) for k in range(self.glevel)]
        self.ObjectTable[self.name].luminosity_gauss[self.index_line] = self.compute_lum_line(self.ObjectTable[self.name].flux_gauss[self.index_line])
        self.ObjectTable[self.name].luminosity_gauss_err[self.index_line] = self.compute_lum_line(self.ObjectTable[self.name].flux_gauss_err[self.index_line])
        self.ObjectTable[self.name].eqw_gauss[self.index_line] = simps(1-(self.fitting_func(self.wave,*self.gauss_pars)+self.continuum_line)/self.continuum_line,self.wave)
        self.ObjectTable[self.name].eqw_gauss_err[self.index_line] = np.sqrt(np.sum(INTEGRAL_ERR*INTEGRAL_ERR))
        self.ObjectTable[self.name].signal_to_noise[self.index_line] = self.ObjectTable[self.name].flux_gauss[self.index_line]/self.noise_level
        self.update_text()
#        self.ax.plot
    
    def update_text(self):
        self.f_text.set_text(r'Flux (integ): $%.4e\ \mathrm{ergs\ s^{-1}cm^{-2}}$'%self.ObjectTable[self.name].flux[self.index_line])
        self.l_text.set_text(r'Lumi (integ): $%.4e\ \mathrm{ergs\ s^{-1}}$'%self.ObjectTable[self.name].luminosity[self.index_line])
        self.e_text.set_text(r'EQW (integ): $%.4f\ \AA$'%self.ObjectTable[self.name].eqw[self.index_line])

        self.fg_text.set_text(r'Flux (gauss): $%.4e\ \mathrm{ergs\ s^{-1}cm^{-2}}$'%self.ObjectTable[self.name].flux_gauss[self.index_line])
        self.lg_text.set_text(r'Lumi (gauss): $%.4e\ \mathrm{ergs\ s^{-1}}$'%self.ObjectTable[self.name].luminosity_gauss[self.index_line])        
        self.fw_text.set_text(r'FWHM (gauss): $%s\ \AA$'%(','.join(['%4f'%self.ObjectTable[self.name].fwhm[self.index_line][i] for i in range(self.glevel)])))
        self.eg_text.set_text(r'EQW (gauss): $%.4f\ \AA$'%self.ObjectTable[self.name].eqw_gauss[self.index_line])
        return None
        
    def compute_lum_line(self,flux):
#        loc_redshift = self.ObjectTable[self.name].line_center[self.index_line][0]/self.rest_frame_line-1
        lum_dist = luminosity_distance(self.redshift)*Mpc*1e5 ##in centimeters
        lum_line = flux*(4*np.pi*lum_dist*lum_dist)
        return lum_line

    def next_galaxy(self,event):
        if self.index_line==len(self.lines)-1:
            self.index+=1
            self.index_line=0
            self.line_shift=0
        else:
            self.index_line+=1
            
        if self.index>=len(self.specnames1D):
            self.index=0

        self.fig.savefig('%s.png'%self.name)

        self.fig.clf()
        
        self.load_spectra(self.index)
        self.interactive_measure_single_line(self.lines[self.index_line],window=self.window)
        
        self.save_to_table()
        return None

    def previous_galaxy(self,event):

        if self.index_line==0:
            self.index-=1
            self.index_line=len(self.lines)-1
            self.line_shift=0
            if self.index<=0:
                self.index=len(self.specnames1D)-1     
        else:
            self.index_line-=1
            
            
        self.fig.savefig('%s.png'%self.name)
        self.fig.clf()

        self.load_spectra(self.index)
        self.interactive_measure_single_line(self.lines[self.index_line],window=self.window)

        self.save_to_table()
        return None

    def save_to_table(self):
        table = open(TABLE_NAME,'w')
        header = "#Name"
        for name in self.linenames:
            header+=" LineCen_%s LineCenErr_%s LineFlux_%s LineLum_%s LineFluxGauss_%s LineFluxGaussErr_%s LineLumGauss_%s LineLumGaussErr_%s LineFWHM_%s LineFWHMErr_%s LineAmplitudes_%s LineAmplitudesErr_%s LineIntegrals_%s LineIntegralsErr_%s EQW_%s EQWGauss_%s EQWGaussErr_%s StoN_%s"%(name,name,name,name,name,name,name,name,name,name,name,name,name,name,name,name,name,name)
        print>> table, header
        for value in self.ObjectTable.values():
            print>>table, value
        return None

    def load_table(self,table_name):
        f=open(table_name,'r')
        txt=f.readlines()
        f.close()

        self.ObjectTable=dict()
        for line in txt:
            if '#' in line:
                continue
            else:
                words=line.split()
                self.ObjectTable[words[0]]=ObjectInfo(words[0],self.linenames,self.linemults)                
                nlines=len(self.linenames)

                nvars = 18                
                ks = np.arange(1,nvars*nlines+1,nvars)                
                
                so = self.ObjectTable[words[0]]
                so.line_center=np.array([[float(a) for a in words[k].translate(None,'[]').split(',')] for k in ks])
                so.line_center_err=np.array([[float(a) for a in words[k].translate(None,'[]').split(',')] for k in ks+nvars-17])
                so.flux=np.array([float(words[k]) for k in ks+nvars-16])
                so.luminosity=np.array([float(words[k]) for k in ks+nvars-15])
                so.flux_gauss=np.array([float(words[k]) for k in ks+nvars-14])
                so.flux_gauss_err=np.array([float(words[k]) for k in ks+nvars-13])
                so.luminosity_gauss=np.array([float(words[k]) for k in ks+nvars-12])
                so.luminosity_gauss_err=np.array([float(words[k]) for k in ks+nvars-11])
                so.fwhm=np.array([[float(a) for a in words[k].translate(None,'[]').split(',')] for k in ks+nvars-10])
                so.fwhm_err=np.array([[float(a) for a in words[k].translate(None,'[]').split(',')] for k in ks+nvars-9])
                so.amplitude=np.array([[float(a) for a in words[k].translate(None,'[]').split(',')] for k in ks+nvars-8])
                so.amplitude_err=np.array([[float(a) for a in words[k].translate(None,'[]').split(',')] for k in ks+nvars-7])
                so.integral=np.array([[float(a) for a in words[k].translate(None,'[]').split(',')] for k in ks+nvars-6])
                so.integral_err=np.array([[float(a) for a in words[k].translate(None,'[]').split(',')] for k in ks+nvars-5])
                so.eqw=np.array([float(words[k]) for k in ks+nvars-4])
                so.eqw_gauss=np.array([float(words[k]) for k in ks+nvars-3])
                so.eqw_gauss_err=np.array([float(words[k]) for k in ks+nvars-2])
                so.signal_to_noise=np.array([float(words[k]) for k in ks+nvars-1])

        return None
        
        
    def change_modes(self,event):
        if event.key=='m':
            self.mode='mask'
        elif event.key=='i':
            self.mode='integrate'
        elif event.key=='c':
            self.mode='continuum'
        elif event.key=='l':
            self.mode='line'
        elif event.key == 'f':
            self.fit_gaussian_to_data()
        elif event.key == 'h':
            self.display_help()
        elif event.key == 'q':
            self.fig.savefig('%s.png'%self.name)
            self.save_to_table()
            self.close('all')
        elif event.key == 'escape':
            self.fig.savefig('%s.png'%self.name)
            self.save_to_table()
            sys.exit()
        elif event.key=='s':
            self.ObjectTable[self.name].nulify()
            self.update_text()
            self.fig.canvas.draw_idle()
        else:
            pass
        self.mode_text.set_text('Mode: %s'%self.mode)
        self.fig.canvas.draw_idle()
    
    def display_help(self):
        fig=mpl.figure(figsize=(10,4))
        fig.suptitle('HELP ON INTERACTIVE OPTIONS')
        fig.text(0.05,0.80,'h - this window')
        fig.text(0.05,0.75,'c - Fit the continumm with a straight line by dragging the mouse between the desired points.')
        fig.text(0.05,0.70,'m - Mask the spectra by dragging the mouse between the points enclosing the region to mask.')
        fig.text(0.05,0.65,'l - Mark the position of the center of the line by clicking on the plot.')
        fig.text(0.05,0.60,'i - Integrate the spectra (using the Simpson method) above the continuum.')
        fig.text(0.05,0.55,'f - Fit a gaussian profile to the displayed data.')
#        fig.text(0.05,0.50,'n - Save current results and move on to the next target.')        
        fig.text(0.05,0.45,'esc - Quit the program.')
        mpl.show(block=False)
        
    def clear_axes(self,event):
        self.fig.clf()
        self.interactive_measure_single_line(self.lines[self.index_line],window=self.window)
        self.fig.canvas.draw_idle()

    def draw_slit(self,eixo,height,width,**kwargs):
        ra_corner = -width/2.
        dec_corner = -height/2.
        R = Rectangle((ra_corner,dec_corner),width,height,fill=False,**kwargs)
        eixo.add_artist(R)
        return None

    def change_contrast(self,val):
        data = self.ax_stamp.get_images()[0]
        data.set_clim(vmin=self.VMIN,vmax=val*(self.VMAX-self.VMIN)+self.VMIN)
        self.fig.canvas.draw_idle()
        return None

    def get_continuum(self,wave,spec,lower,upper):
        k1 = np.where(wave>lower)[0][0]
        k2 = np.where(wave>upper)[0][0]
        x,y=np.median(wave[k1:k2]),np.median(spec[k1:k2])
        return x,y
        
    def estimate_continuum(self,wave,spec):
        x1,y1 = self.get_continuum(wave,spec,self.limbl-self.spec_resol,self.limbl+self.spec_resol)
        x2,y2 = self.get_continuum(wave,spec,self.limrd-self.spec_resol,self.limrd+self.spec_resol)
        
        slope = (y2-y1)/(x2-x1)
        yint = y2-slope*x2
        
        self.point_contblue.set_data([x1],[y1])
        self.point_contred.set_data([x2],[y2])
        
        return slope*wave+yint        
        
    def interactive_measure_single_line(self,rest_frame_line,window=100):

        self.start_buttons(self.fig)

        self.gen_ax = self.fig.add_axes([0.22,0.70,0.73,0.15])
        self.ax_2D = self.fig.add_axes([0.22,0.85,0.73,0.10],sharex=self.gen_ax)
        self.ax = self.fig.add_axes([0.22,0.15,0.73,0.55])
        self.ax_2Dzoom = self.fig.add_axes([0.22,0.05,0.73,0.1],sharex=self.ax)

        self.ax_slider = self.fig.add_axes([0.055,0.59,0.13,0.02])
        smooth_slid=self.fig.add_axes([0.055,0.56,0.13,0.02])
        self.ax_stamp = self.fig.add_axes([0.02,0.68,0.17,0.30])
                
        self.name = 'P%i_Q%i_%03i'%(self.table['Pointing'][self.index],self.table['Quadrant'][self.index],self.table['ObjID'][self.index])
        self.redshift = self.table['specZ'][self.index]
        self.wave_line = rest_frame_line*(1+self.redshift)
        self.rest_frame_line = rest_frame_line
        self.glevel = self.linemults[self.index_line]

        try:
            self.ObjectTable[self.name]
        except KeyError:
            self.ObjectTable[self.name]=ObjectInfo(self.name,self.linenames,self.linemults)
            
        if (self.wave_line < self.wavelength[0]+window/2.0) or (self.wave_line > self.wavelength[-1]-window/2.0):
            print("Skipping due to lack of spectral coverage: z=%.3f"%(self.redshift))
            self.save_to_table()
            self.fig.clf()
            if self.index_line==len(self.lines)-1:
                self.index+=1
                self.index_line=0
            else:
                self.index_line+=1
                
            if self.index>=len(self.specnames1D):
                self.index=0
            
            self.load_spectra(self.index)
            self.interactive_measure_single_line(self.lines[self.index_line],window=self.window)

#==============================================================================
# IMAGE DEFINITION        
#==============================================================================
        self.VMIN=-0.005
        self.VMAX=0.034
        image_data = self.stampI
        N,M=image_data.shape
        hsize_arcsec = N/2*self.pixscale
        self.ax_stamp.imshow(image_data,extent=(-hsize_arcsec,hsize_arcsec,-hsize_arcsec,hsize_arcsec),cmap=self.COLORMAP,vmin=self.VMIN,vmax=self.VMAX,aspect='equal')
        self.draw_slit(self.ax_stamp,8,1,color=self.SLITCOLOR,linewidth=1.5)
#        self.ax_stamp.set_xticks([])
#        self.ax_stamp.set_yticks([])
        self.ax_stamp.set_xlabel(r'$\Delta \alpha\ [{}^{\prime\prime}]$',fontsize=15)
        self.ax_stamp.set_ylabel(r'$\Delta \delta\ [{}^{\prime\prime}]$',fontsize=15)
        self.ax_stamp.tick_params(labelleft='off',labelright='on',labelbottom='on')

        step=0.8
        self.ax_stamp.arrow(-3.0,3.0,0,step,color='w',head_width=0.1)
        self.ax_stamp.arrow(-3.0,3.0,-step-0.1,0,color='w',head_width=0.1)
        self.ax_stamp.text(-3.0,4.0,'N',va='bottom',ha='center',fontsize=9,color='white')
        self.ax_stamp.text(-4.1,3.0,'E',va='center',ha='right',fontsize=9,color='white')

        
        self.slide_scale = mpw.Slider(self.ax_slider, 'Contrast', 0.0, 1.0, valinit=1.0,closedmin=False,color=self.SLIDERCOLOR)
        self.slide_scale.on_changed(self.change_contrast)

  
#==============================================================================
#       SPECTRAL RANGE  
#==============================================================================
        wave_start= self.wave_line-window/2.0
        wave_end  = self.wave_line+window/2.0
        
        k1 = np.where(self.wavelength>wave_start)[0][0]
        k2 = np.where(self.wavelength>wave_end)[0][0]

        self.kstart=k1
        self.kend=k2
        self.window_size=window

#==============================================================================
# GENERAL 1D and 2D        
#==============================================================================
        self.ax_2D.imshow(np.pad(self.spec2D,((5,5),(0,0)),'constant'),cmap=self.COLORMAP,extent=(self.wavelength[0],self.wavelength[-1],0,self.spec2D.shape[0]+10),aspect='auto',vmin=-0.0025,vmax=0.01)#,interpolation='none')
        self.ax_2D.set_yticks([])


#        self.gen_ax.plot(self.wavelength,self.spec1D,'k-')

        self.gen_ax.plot(self.wavelength,self.spec1D,'k-',drawstyle='steps-mid')
        draw_lines(self.gen_ax,self.redshift,self.wavelength,self.spec1D,9)
        self.gen_ax.fill_betweenx([0.95*np.amin(self.spec1D),1.05*np.amax(self.spec1D)],wave_start,wave_end,color='Pink',alpha=0.75,zorder=-200)
        specmin=0.95*np.amin(max(0,np.amin(self.spec1D[np.isnan(self.spec1D)==False])))
        specmax=1.05*np.amax(self.spec1D[np.isnan(self.spec1D)==False])
        self.gen_ax.imshow(self.OHintensity_masked,extent=(self.OHwave[0]-self.delta_OHwave,self.OHwave[-1]+self.delta_OHwave,specmin,specmax),aspect='auto',zorder=-100,cmap=self.OHcmap,vmax=8,vmin=-1)        

        self.gen_ax.set_ylim(specmin,specmax)
        self.gen_ax.set_xlim(self.wavelength[0],self.wavelength[-1])
        self.gen_ax.set_xticks([])
        self.gen_ax.set_yticks([])
    #    ax.plot(wave[k1:k2],spectra[k1:k2],'k-')
 
#==============================================================================
# ZOOM 1D and 2D       
#==============================================================================
        self.wave_gauss= np.linspace(self.wavelength[k1],self.wavelength[k2-1],num=10000)
#        self.func_spec = interp1d(self.wavelength[k1:k2],self.spec1D[k1:k2],kind='linear')
        self.wave=self.wavelength[k1:k2].copy()
        self.spec = self.spec1D[k1:k2].copy()
        
        self.specline,=self.ax.plot(self.wavelength[k1:k2],self.spec1D[k1:k2],'-',color=self.SPECCOLOR,linewidth=2)
        specmin=np.amin(self.spec1D[k1:k2])
        specmin = (1-np.sign(specmin)*0.05)*specmin
        specmax=1.05*np.amax(self.spec1D[k1:k2])
        
        #        self.specline,=self.ax.plot(self.wavelength[k1:k2],self.spec1D[k1:k2],'b-',drawstyle='steps-mid')
        self.ax.vlines(self.wave_line,0.9*np.amin(self.spec1D[k1:k2]),1.1*np.amax(self.spec1D[k1:k2]),'Plum','-',linewidth=3,zorder=-5)
        self.ax.vlines(self.wave_line+self.line_shift,0.9*np.amin(self.spec1D[k1:k2]),1.1*np.amax(self.spec1D[k1:k2]),'SkyBlue','-',linewidth=3,zorder=-5)
        self.ax.imshow(self.OHintensity_masked,extent=(self.OHwave[0]-self.delta_OHwave,self.OHwave[-1]+self.delta_OHwave,specmin,specmax),aspect='auto',zorder=-100,cmap=self.OHcmap,vmax=8,vmin=-1)        
##        self.ax.vlines(self.OHwave[self.OHintensity[0,:]>self.sky_threshold],specmin,specmax,'Gray','-',linewidth=0.5,zorder=-5)
        
        self.limbl = self.wave_line-window/3.0
        self.limrd = self.wave_line+window/3.0
        self.active_contline,= self.ax.plot([0,0],[-99,-99])
        self.active_cont_center = self.wave_line
        self.point_contblue,= self.ax.plot([],[],'x',color=self.CONTPOINTCOLOR,ms=20,mew=3)
        self.point_contred,= self.ax.plot([],[],'x',color=self.CONTPOINTCOLOR,ms=20,mew=3)

        self.cont_intblue, = self.ax.plot([self.limbl-self.spec_resol,self.limbl+self.spec_resol],[(1+np.sign(specmin)*0.55)*specmin,(1+np.sign(specmin)*0.55)*specmin],'-',color='blue',lw=3)
        self.cont_intred, = self.ax.plot([self.limrd-self.spec_resol,self.limrd+self.spec_resol],[(1+np.sign(specmin)*0.55)*specmin,(1+np.sign(specmin)*0.55)*specmin],'-',color='red',lw=3)
        self.continuum_line = self.estimate_continuum(self.wavelength[k1:k2],self.spec1D[k1:k2])
        
        self.contline,=self.ax.plot(self.wavelength[k1:k2],self.continuum_line,color=self.CONTCOLOR,ls='-')
        self.gaussline,=self.ax.plot(self.wave_gauss,np.ones(np.size(self.wave_gauss)),color=self.GAUSSCOLOR,ls='--',linewidth=2)        
        self.ax.tick_params(labelleft='off',labelright='on',labelbottom='off')

        dummy, cont_spectra = mask_spectra(self.redshift,self.wavelength[k1:k2],self.spec1D[k1:k2],0,slim=0,kmask=12)
        self.noise_level = np.nanstd(cont_spectra)
        self.ax.fill_between(self.wavelength[k1:k2],np.median(cont_spectra)-self.noise_level,np.median(cont_spectra)+self.noise_level,color=self.NOISECOLOR,zorder=-10,alpha=0.5)
        
        self.wave_mark=self.ax.vlines(self.wave_line+self.line_shift,specmin,specmax,self.WAVEMARKCOLOR,'--',linewidth=2)
        if 'OII' in self.linenames[self.index_line]:
            self.wave_mark2 = self.ax.vlines((self.rest_frame_line+deltaOII)*(1+self.redshift)+self.line_shift,specmin,specmax,self.WAVEMARKCOLOR,'--',linewidth=2)

        self.ax.text(self.wave_line+window/50.0,np.amax(self.spec1D[k1:k2]),self.linenames[self.index_line],va='top',ha='left')
        self.mode='continuum'

        self.ax.set_ylabel(r'$F_\lambda\ [\mathrm{10^{%i}ergs\ s^{-1}cm^{-2}\AA^{-1}}]$'%(np.log10(self.normflux)),fontsize=12)
##        self.ax.yaxis.set_label_position("right")
        self.ax.minorticks_on()
        self.ax.set_ylim(specmin,specmax)
        self.ax.set_xlim(self.wave[1],self.wave[-2])
        
        
        self.ax_2Dzoom.imshow(self.spec2D[:,self.kstart:self.kend],cmap=self.COLORMAP,extent=(self.wavelength[k1],self.wavelength[k2],0,self.spec2D.shape[0]),aspect='auto',vmin=-0.0025,vmax=0.01)#,interpolation='none')
        self.ax_2Dzoom.set_yticks([])
        self.ax_2Dzoom.set_xlabel(r'$\lambda\ [\mathrm{\AA}]$',fontsize=15,labelpad=-15)

#==============================================================================
#         BUTTONS AND SLIDERS
#==============================================================================

        resbut = self.fig.add_axes([0.015,0.05,0.1,0.05])
        self.resbut=mpw.Button(ax=resbut,label='Reset Spectra',color=self.BUTTONCOLOR,hovercolor='Gold')
        self.resbut.on_clicked(self.reset_data)

        clearbut = self.fig.add_axes([0.115,0.05,0.1,0.05])
        self.clearbut=mpw.Button(ax=clearbut,label='Clear Axes',color=self.BUTTONCOLOR,hovercolor='Gold')
        self.clearbut.on_clicked(self.clear_axes)

        nextbut = self.fig.add_axes([0.115,0.10,0.1,0.05])
        self.nextbut=mpw.Button(ax=nextbut,label='Next',color=self.BUTTONCOLOR,hovercolor='Gold')
        self.nextbut.on_clicked(self.next_galaxy)

        prevbut = self.fig.add_axes([0.015,0.10,0.1,0.05])
        self.prevbut=mpw.Button(ax=prevbut,label='Previous',color=self.BUTTONCOLOR,hovercolor='Gold')
        self.prevbut.on_clicked(self.previous_galaxy)

        self.slider_smooth = matplotlib.widgets.Slider(smooth_slid,'Smooth', 0.0, 3.0, valinit=0.0,color=self.SLIDERCOLOR,closedmin=True)
        self.slider_smooth.on_changed(self.smooth_1Dspectra)

#==============================================================================
#  TEXT CREATION        
#==============================================================================
        self.mode_text = self.fig.text(0.09,0.45,'Mode: %s'%self.mode,ha='center')

        self.f_text = self.fig.text(0.01,0.40, r'Flux (integ): $%.4e\ \mathrm{ergs\ s^{-1}cm^{-2}}$'%self.ObjectTable[self.name].flux[self.index_line],fontsize=11)
        self.l_text = self.fig.text(0.01,0.37, r'Lumi (integ): $%.4e\ \mathrm{ergs\ s^{-1}}$'%self.ObjectTable[self.name].luminosity[self.index_line],fontsize=11)
        self.e_text = self.fig.text(0.01,0.34, r'EQW (integ): $%.4f\ \AA$'%self.ObjectTable[self.name].eqw[self.index_line],fontsize=11)
        self.fg_text = self.fig.text(0.01,0.31, r'Flux (gauss): $%.4e\ \mathrm{ergs\ s^{-1}cm^{-2}}$'%self.ObjectTable[self.name].flux_gauss[self.index_line],fontsize=11)
        self.lg_text = self.fig.text(0.01,0.28, r'Lumi (gauss): $%.4e\ \mathrm{ergs\ s^{-1}}$'%self.ObjectTable[self.name].luminosity_gauss[self.index_line],fontsize=11)
        self.fw_text = self.fig.text(0.01,0.25,r'FWHM (gauss): $%s\ \AA$'%(','.join(['%4f'%self.ObjectTable[self.name].fwhm[self.index_line][i] for i in range(self.glevel)])),fontsize=11)
        self.eg_text = self.fig.text(0.01,0.22, r'EQW (gauss): $%.4f\ \AA$'%self.ObjectTable[self.name].eqw_gauss[self.index_line],fontsize=11)

        
#==============================================================================
#   CANVAS CONNECTION AND SHOW
#==============================================================================
        self.fig.canvas.mpl_connect('button_press_event',self.get_button_press)       
        self.fig.canvas.mpl_connect('button_release_event',self.get_button_release)
        self.fig.canvas.mpl_connect('key_press_event',self.change_modes)
        self.check_buttons.on_clicked(self.drawstyle)

        self.ax_2D.set_title(str(self.index+1)+' : '+self.name+" - Click 'h' for a list of available options.")
#       self.fig.canvas.mpl_connect('motion_notify_event', get_mouse_position)
        

        self.toggle_drawstyle()
        self.fig.canvas.draw_idle()
        



    def smooth_1Dspectra(self,val):
        self.specline.set_data([self.wave,snd.gaussian_filter(self.spec,val)])                
        self.fig.canvas.draw_idle()
    
    def toggle_drawstyle(self):
        if self.step_style is True:
            self.specline.set_drawstyle('steps-mid')
        else:
            self.specline.set_drawstyle('default')
        self.fig.canvas.draw_idle()
        
    def drawstyle(self,var):
        self.step_style = not self.step_style
        self.toggle_drawstyle()
        
    def start_buttons(self,figura):

        checkax = figura.add_axes([0.07,0.45, 0.075, 0.15])
        self.check_buttons = mpw.CheckButtons(checkax, ['Step-Plot'],[self.step_style])
        
        for cross in self.check_buttons.lines:
            for line in cross:
                line.set_color(self.BUTTONCOLOR)
                line.set_linewidth(1.0)
        
        for rect in self.check_buttons.rectangles:
            rect.set_edgecolor(self.BUTTONCOLOR)
            rect.set_linewidth(1.5)
            
        for side in ['bottom', 'top','left','right']:
            self.check_buttons.ax.spines[side].set_visible(False)

        return None
##SAMPLE_TABLE = np.loadtxt('spectra-secure',dtype={'names':('Pointing', 'Quadrant', 'ObjID', 'SlitID', 'RAwrong', 'DECwrong', 'RA', 'DEC', 'specZ', 'Flag')\
##                                                ,'formats':('i4','i4','i4','i8','f8','f8','f8','f8','f8','i4')})





if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Line Measurement Interface.")
    parser.add_argument('-l','--line',metavar='NAME',type=str,help="Set of lines on which to measure, separated by commas.")
    parser.add_argument('-m','--multiplicity',metavar='NAME',type=str,help="Number of gaussians to fit the lines, separated by commas.")
    parser.add_argument('-c','--catalog',metavar='NAME',type=str,default='spectra-secure.fits',help="Catalog on which to run the interface.")
    parser.add_argument('-i','--name',metavar='NAME',type=int,default=0,help="Name of the first galaxy to start with")
    parser.add_argument('-d','--draw',action='store_true',help="If present replots the figures, if not runs ellipse and then replots the figures.")

    args = parser.parse_args()

    SAMPLE_TABLE = pyfits.open(args.catalog)[1].data

    VISUALCHECK=args.draw

    index=args.name
        
    
    line_names = args.line.split(',')
    line_centers = [W[L==line][0] for line in line_names]
    deltaOII=2.79 ## doblet separation

    multiplicity = [int(m) for m in args.multiplicity.split(',')]
                    
    assert len(line_names)==len(multiplicity),'Number of given lines must be equal to multiplicity numbers'
    
#    print line_names,line_centers,multiplicity

    clean_names = [line.translate(None,r'$\\[]') for line in line_names]           
    TABLE_NAME='VIMOS_LSS_line_fluxes_interactive_%s.txt'%(''.join(clean_names))

    A = SimpleWidget(SAMPLE_TABLE,index,line_centers,line_names,multiplicity)
