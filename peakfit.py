# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 10:09:11 2018

@author: johan
"""
import glob
import numpy as np
import os
from numpy import exp, linspace, random,  pi, sqrt
from lmfit import Model
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel

#def gaussian(x, amp, cen, wid):
#    return amp * exp(-(x-cen)**2 / wid)
def trim_spectra(spectra, start = None, end = None):

    if start is not None:
        smaller_than_start = np.where(spectra[0] < start)        
        spectra = np.delete(spectra, smaller_than_start, axis=1)
    if end is not None:
        larger_than_end  = np.where(spectra[0] > end)
        spectra = np.delete(spectra, larger_than_end, axis=1)
    
    return spectra

data_folder = 'C:/Syncthing/Teslafir_data/2017/092_CoNb2O6_rot_Voigt/Eval/Second_run/Bdep/2K5/sp10755/pol8325'

os.chdir(data_folder)
name_list = glob.glob("*.dat")
#    name_list = [x for x in name_list if not x in exclude]
datalist = [np.loadtxt(filename, unpack=True) for filename in name_list]
data = datalist[13]

fig = plt.figure('fit')
fig.clf()
ax = fig.add_subplot(111)
#

gmodel1 = GaussianModel(prefix='g1_')
fits = []
# for data in datalist:
data = trim_spectra(data, start=3, end=40)
x = data[0]
y = data[1]
pars1 = gmodel1.guess(y, x=x)
# pars1['g1_center'].set(14, min=10, max=15)
# pars1['g1_amplitude'].set(min=0, max=200)
# pars1['g1_fwhm'].set(max=10)
# pars1['g1_sigma'].set(max=20)

gmodel2 = GaussianModel(prefix='g2_')
pars2 = gmodel2.guess(y, x=x)
# pars2['g2_center'].set(26, min=25, max=29)
# pars2['g2_amplitude'].set(min=0, max=200)
# pars2['g2_fwhm'].set(max=10)
# pars2['g2_sigma'].set(max=20)
model = gmodel1 + gmodel2


pars = pars1 + pars2



init = model.eval(pars, x=x)
out = model.fit(y, pars, x=x)
fits.append(out.best_fit)

# print(out.fit_report())

# plt.plot(x, y, 'b')
# plt.plot(x, init, 'k--')
plt.plot(x, out.best_fit)
# plt.xlim(3,40)
# plt.ylim(-5,40)



plt.show()