#!/usr/bin/env python
# coding: utf-8

# In[21]:


# colour_system.py
import numpy as np
from scipy.constants import h, c, k
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
import pylab as pl
from scipy import interpolate
import tmm as tmm


# In[22]:


##Some functions we will use later.
##These functions are from the website.

def xyz_from_xy(x, y):
    """Return the vector (x, y, 1-x-y)."""
    return np.array((x, y, 1-x-y))

class ColourSystem:
    """A class representing a colour system.

    A colour system defined by the CIE x, y and z=1-x-y coordinates of
    its three primary illuminants and its "white point".

    TODO: Implement gamma correction

    """

    # The CIE colour matching function for 380 - 780 nm in 5 nm intervals
    cmf = np.loadtxt('cie-cmf.txt', usecols=(1,2,3))

    def __init__(self, red, green, blue, white):
        """Initialise the ColourSystem object.

        Pass vectors (ie NumPy arrays of shape (3,)) for each of the
        red, green, blue  chromaticities and the white illuminant
        defining the colour system.

        """

        # Chromaticities
        self.red, self.green, self.blue = red, green, blue
        self.white = white
        # The chromaticity matrix (rgb -> xyz) and its inverse
        self.M = np.vstack((self.red, self.green, self.blue)).T 
        self.MI = np.linalg.inv(self.M)
        # White scaling array
        self.wscale = self.MI.dot(self.white)
        # xyz -> rgb transformation matrix
        self.T = self.MI / self.wscale[:, np.newaxis]

    def xyz_to_rgb(self, xyz, out_fmt=None):
        """Transform from xyz to rgb representation of colour.

        The output rgb components are normalized on their maximum
        value. If xyz is out the rgb gamut, it is desaturated until it
        comes into gamut.

        By default, fractional rgb components are returned; if
        out_fmt='html', the HTML hex string '#rrggbb' is returned.

        """

        rgb = self.T.dot(xyz)
        if np.any(rgb < 0):
            # We're not in the RGB gamut: approximate by desaturating
            w = - np.min(rgb)
            rgb += w
        if not np.all(rgb==0):
            # Normalize the rgb vector
            rgb /= np.max(rgb)

        if out_fmt == 'html':
            return self.rgb_to_hex(rgb)
        return rgb

    def rgb_to_hex(self, rgb):
        """Convert from fractional rgb values to HTML-style hex string."""

        hex_rgb = (255 * rgb).astype(int)
        return '#{:02x}{:02x}{:02x}'.format(*hex_rgb)

    def spec_to_xyz(self, spec):
        """Convert a spectrum to an xyz point.

        The spectrum must be on the same grid of points as the colour-matching
        function, self.cmf: 380-780 nm in 5 nm steps.

        """

        XYZ = np.sum(spec[:, np.newaxis] * self.cmf, axis=0)
        den = np.sum(XYZ)
        if den == 0.:
            return XYZ
        return XYZ / den

    def spec_to_rgb(self, spec, out_fmt=None):
        """Convert a spectrum to an rgb value."""

        xyz = self.spec_to_xyz(spec)
        return self.xyz_to_rgb(xyz, out_fmt)

illuminant_D65 = xyz_from_xy(0.3127, 0.3291)
cs_hdtv = ColourSystem(red=xyz_from_xy(0.67, 0.33),
                       green=xyz_from_xy(0.21, 0.71),
                       blue=xyz_from_xy(0.15, 0.06),
                       white=illuminant_D65)

cs_smpte = ColourSystem(red=xyz_from_xy(0.63, 0.34),
                        green=xyz_from_xy(0.31, 0.595),
                        blue=xyz_from_xy(0.155, 0.070),
                        white=illuminant_D65)

cs_srgb = ColourSystem(red=xyz_from_xy(0.64, 0.33),
                       green=xyz_from_xy(0.30, 0.60),
                       blue=xyz_from_xy(0.15, 0.06),
                       white=illuminant_D65)


cs = cs_hdtv


# In[ ]:





# In[ ]:





# In[23]:


##Real part of the refractive index for the si

a3 =[208.219, 218.493, 227.397, 234.247, 247.945, 261.644, 267.808, 271.918, 274.658, 276.712, 278.767, 280.822, 282.192, 284.247, 285.616, 286.986, 288.356, 289.726, 291.096, 293.151, 294.521, 297.26, 306.164, 322.603, 335.616, 345.89, 351.37, 355.479, 358.219, 360.274, 363.014, 365.068, 373.973, 379.452, 382.877, 386.301, 390.411, 395.89, 400.685, 407.534, 415.068, 423.973, 434.932, 445.89, 458.904, 472.603, 487.671, 502.74, 519.178, 534.932, 551.37, 567.808, 583.562, 600.0, 616.438, 632.877, 649.315, 666.438, 682.877, 699.315, 716.438, 732.877, 750.0, 766.438, 782.877, 800.0, 816.438]

b3 =[1.0, 1.15323, 1.33065, 1.51613, 1.56452, 1.67742, 1.8629, 2.05645, 2.25806, 2.45161, 2.65323, 2.85484, 3.05645, 3.25806, 3.45968, 3.66129, 3.85484, 4.05645, 4.25806, 4.45968, 4.66129, 4.8629, 5.01613, 5.03226, 5.15323, 5.32258, 5.50806, 5.70968, 5.90323, 6.10484, 6.30645, 6.5, 6.67742, 6.52419, 6.32258, 6.12903, 5.93548, 5.74194, 5.54839, 5.3629, 5.18548, 5.00806, 4.85484, 4.70968, 4.57258, 4.45968, 4.35484, 4.27419, 4.19355, 4.12903, 4.07258, 4.02419, 3.97581, 3.93548, 3.90323, 3.87097, 3.83871, 3.81452, 3.79839, 3.77419, 3.75806, 3.74194, 3.72581, 3.70968, 3.70161, 3.68548, 3.66935]

pl.plot(a3,b3,"ro")

for kind in ["cubic"]:#The interpolation method is "Cubic method".
    
    f3=interpolate.interp1d(a3,b3,fill_value="extrapolate")  #define a function named f3, which is the real part of the refractive index for the Si.
    # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
    xnew=a3
    ynew=f3(a3)
    pl.plot(xnew,ynew,label=str(kind))
pl.legend(loc="lower right")
plt.xlabel('wavelength (in nm)')
plt.title('Real part of refractive index for Si')
pl.show()


# In[24]:


##Extinctive part of the refractive index for the si (k)

a4 =[206.944, 215.278, 222.917, 231.25, 243.75, 250.0, 254.167, 258.333, 261.806, 264.583, 267.361, 270.139, 272.222, 275.0, 277.778, 281.944, 289.583, 291.667, 293.056, 295.139, 296.528, 298.611, 300.0, 302.083, 304.167, 307.639, 311.111, 315.278, 321.528, 329.861, 340.972, 356.25, 361.806, 365.278, 367.361, 368.75, 370.139, 371.528, 373.611, 375.0, 376.389, 377.778, 380.556, 383.333, 386.111, 390.278, 397.917, 408.333, 422.917, 438.889, 455.556, 472.222, 488.889, 505.556, 522.917, 540.278, 556.944, 574.306, 591.667, 609.028, 626.389, 643.75, 661.111, 677.778, 695.139, 712.5, 729.861, 747.222, 764.583, 781.944, 799.306, 816.667]


b4 =[2.90411, 3.04795, 3.20548, 3.35616, 3.42466, 3.58219, 3.74658, 3.91781, 4.08219, 4.25342, 4.41781, 4.58904, 4.76027, 4.93151, 5.09589, 5.26712, 5.31507, 5.15068, 4.97945, 4.80822, 4.63699, 4.46575, 4.29452, 4.12329, 3.9589, 3.78767, 3.61644, 3.45205, 3.29452, 3.14384, 3.0137, 2.9863, 2.82877, 2.65753, 2.49315, 2.32192, 2.15068, 1.97945, 1.80822, 1.63699, 1.46575, 1.29452, 1.13014, 0.9589, 0.78767, 0.62329, 0.46575, 0.33562, 0.24658, 0.19178, 0.15753, 0.12329, 0.09589, 0.08219, 0.07534, 0.06164, 0.05479, 0.04795, 0.04795, 0.0411, 0.0411, 0.03425, 0.03425, 0.0274, 0.0274, 0.0274, 0.0274, 0.0274, 0.0274, 0.0274, 0.0274, 0.02055]

pl.plot(a4,b4,"ro")

for kind in ["cubic"]:#The interpolation method is "Cubic method".
    
    f4=interpolate.interp1d(a4,b4,fill_value="extrapolate")
    # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)

    ynew=f4(a4)
    pl.plot(a4,ynew,label=str(kind))
pl.legend(loc="lower right")
plt.xlabel('wavelength (in nm)')
plt.title('imaginary part of refractive index for Si')
pl.show()


# In[25]:


##In this section, we will combine the material and the substrate together, to form a 2D device. And calculate the 
##reflectance of the whole system.
#This function will return a function f5, which is the function of the reflectance, by given the thickness of SiO2.


def reflec_absorp_func(dSiO2,th_0, pol, lam_max, lam_min):
    
    """d_list is the thickness of the material, dSiO2 is the thickness of the SiO2 layer, th_0 is the incident angle,
    pol is the polarization of incident light, lam_max and lam_min are the ranges of the incident light. """
                 
    d_list =[np.inf,dSiO2,np.inf]
    n_SiO2 = 1.479                           ##This is the permitivity of the silicon. It's a constant!
    
    mu=1                                           ### Suppose the relative permeability is ONE for all layers
    
  
    lam_range = np.linspace(lam_min,lam_max,num=500)
    

    reflectance = []
    for lam_vac in lam_range:
        
        energy = 1240/lam_vac
        
        
        n_si = complex(f3(lam_vac),f4(lam_vac))          ##input the refractive index for silicon.
        
      
        n_list =[1, n_SiO2, n_si]
        
        coh_tmm_data = tmm.tmm_core.coh_tmm(pol, n_list, d_list, th_0, lam_vac)
        r = coh_tmm_data['r']
        reflectance.append(tmm.R_from_r(r))
        
    absorption = 1-np.array(reflectance)    
    h = 6.62607004*1e-34
    E = h*3*1e8/(lam_range*1e-9*1.602176634*1e-19) ### in eV
    E = E.tolist()
    

    
    for kind in ["cubic"]:
        f5=interpolate.interp1d(E,reflectance,fill_value="extrapolate")
        
        """f5 is the function of the reflectance for MoS2o"""
        # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
        
        xnew=E
        ynew=f5(E)
        
        
    ##The folowing few lines can draw the plot of the caculated refelctance.
    ##pl.plot(xnew,ynew,label=str(kind))
   
    #plt.xlabel('E(in eV)')
    #plt.ylabel('reflectance')
    #plt.title('relationship between reflectance and photon energy(%s'%(pol)+' polarized)')
    #pl.show()

    
    #plt.figure()
    #plt.plot(E, absorption)
    #plt.xlabel('E(in eV)')
    #plt.ylabel('absorption')
    #plt.title('relationship between absorption and photon energy(%s'%(pol)+' polarized)')
    #plt.show() 
    return f5


# In[26]:


#This function will return a value re, which is the reflectance at wavelength lam, with thickness of SiO2 as d.

def reflectance(lam,d):
    x = 1240/lam                                           ##Convert the wavelength to the energy.
    r = reflec_absorp_func(d, 0, 'p', 330, 800)    ##The range of the wavelength must include the range of the visible light.
    re = r(x)
    return re


# In[27]:


#This function will give back the radiation power, at wavelength lam, at temperature T, with the thickness of SiO2 as d.
def planck(lam, T, d):

    re = reflectance(lam,d)
    lam_m = lam / 1.e9
    fac = h*c/lam_m/k/T
    B = (2*h*c**2)/(lam_m**5)/(np.exp(fac) - 1)*re  ##Since we are calculating the reflected power, we need a square here. 
    return B


# In[42]:




##The color of MoS_2 with the substrate at different thicknesses.(from 280 nm to 560 nm)

fig, ax = plt.subplots()

# The grid of visible wavelengths corresponding to the grid of colour-matching
# functions used by the ColourSystem instance.
lam = np.arange(380., 781., 5)

for i in range(205):
   
    T =  2500
    
    d= 5*i
    # Calculate the black body spectrum and the HTML hex RGB colour string
    # it looks like
    spec = planck(lam, T, d)
    html_rgb = cs.spec_to_rgb(spec, out_fmt='html')
    
    # Place and label a circle with the colour of a black body at temperature T
    x, y = (i % 205), -(i // 205)
    rectangle = Rectangle((x, y*1.2), 0.5, 35, fc=html_rgb)
    ax.add_patch(rectangle)
    if d%200==0:
        ax.annotate('{:2d} nm'.format(d), xy=(x, y*1.2-2.5), va='center',
                    ha='center', color=html_rgb)

# Set the limits and background colour; remove the ticks
ax.set_xlim(-10,215)
ax.set_ylim(-10, 40)
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor('k')
plt.show()


# In[ ]:




