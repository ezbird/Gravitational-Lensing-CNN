#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
Cut the big 10GB COSMOS2020 mosaic file (I have i and g filter) into smaller pieces so we can see them better with DS9.
Should be able to to use astropy Cutout2D
"""
import numpy as np 
from matplotlib import pyplot as plt 
import h5py # used in the Data Visualization section 

from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.io import fits,ascii,votable
from astropy import units as u 
from astropy import constants as const
from astropy import table
from astropy.cosmology import Planck15,FlatLambdaCDM
from astropy.nddata.utils import Cutout2D
from astropy.wcs import WCS

# Look at this in the i band
fits_file = '/home/dobby/LensingProject/COSMOS2020_complete_catalog/HSC_i_SSP_PDR2_19_07_19_v3.fits'
band = "i"

# Look at this in the g band
#fits_file = '/home/dobby/LensingProject/COSMOS2020_complete_catalog/HSC_G_SSP_PDR2.fits'
#band = "g"

# ----------------------------------------------------------------------

#read in the fits file
#insciim = fits.open(fits_file)
data = fits.getdata(fits_file)

#Read in the header
hdr = fits.getheader(fits_file)

x=1
size = (100, 100) # size I want the image to be in pixels
wcs1 = WCS(hdr)

f=open("/home/dobby/LensingProject/coordinates_of_47_secondary.txt","r")
lines=f.readlines()
RAs=[]
DECs=[]
for x in lines:
    RAs.append(x.split(' ')[0])
for x in lines:
    DECs.append(x.split(' ')[1])
f.close()

for i in range(len(RAs)):
     ra  = float(RAs[i].replace('\n',""))
     dec = float(DECs[i].replace('\n',""))
     print(ra)
     cut_test = Cutout2D(data, (SkyCoord(ra*u.deg,dec*u.deg)), size, wcs=wcs1)

     img_hdu = fits.PrimaryHDU(cut_test.data, header=hdr)
     fits_to_write = "/home/dobby/LensingProject/"+band+"_band_fits_secondary/cutout_"+str(ra)+"_"+str(dec)+"_"+band+".fits"
     img_hdu.writeto(fits_to_write, overwrite=True)

print('Done.')
