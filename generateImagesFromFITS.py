import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
import os 
plt.style.use(astropy_mpl_style)


# Best 20
base_g = '/home/dobby/LensingProject/g_band_fits_best/'
base_i = '/home/dobby/LensingProject/i_band_fits_best/'
# Secondary 47
#base_g = '/home/dobby/LensingProject/g_band_fits_secondary/'
#base_i = '/home/dobby/LensingProject/i_band_fits_secondary/'

# Makes list of all files in this directory
# Get all FITS files in this way.
images_g = os.listdir(base_g)
images_i = os.listdir(base_i)

vmin = 0
vmax = 0.002


for i in range(len(images_i)):
     print(i)
     image_file_g = base_g + images_g[i]
     image_file_i = image_file_g.replace("_g", "_i").replace("g_", "i_")

     print("image_file_g",image_file_g)
     print("image_file_i",image_file_i)
     print("-------------")
     ##############################################################################
     #fits.info(image_file_g) # display the structure of the file:
     #continue
     ##############################################################################
     # Generally the image information is located in the Primary HDU, also known
     # as extension 0. Here, we use `astropy.io.fits.getdata()` to read the image
     # data from this first extension using the keyword argument ``ext=0``:
     
     image_data_g = fits.getdata(image_file_g)#, ext=5)
     image_data_i = fits.getdata(image_file_i)#, ext=9)

     '''    
     # Make green filter image     
     plt.figure()
     plt.imshow(image_data_g, cmap='gray')
     #plt.colorbar()
     #plt.title("g: "+images[i], fontsize=10)
     plt.axis('off')
     plt.savefig('/home/dobby/Lensing_Project/COSMOS2020_cutouts/training_set/i_band_png/' + output + str(i) + '_i.png', bbox_inches='tight', dpi=300)#, vmin=vmin,vmax=vmax)
     plt.clf()

    
     image_file = base + images[i]
     fits.info(image_file) # display the structure of the file:
     image_data_i = fits.getdata(image_file, ext=9)

     # Make infrared filter image
     plt.figure()
     plt.imshow(image_data_i, cmap='gray')
     #plt.colorbar()
     #plt.title("i: "+images[i], fontsize=10)
     
     plt.savefig('/home/dobby/Lensing Project/png_training_set/' + output + str(i) + '_i.png', bbox_inches='tight', dpi=300, vmin=vmin,vmax=vmax)
     plt.clf()

     '''

     # Make green minus infrared image
     image_subtracted = image_data_g - image_data_i
     
     print(image_subtracted)
     print(image_subtracted.shape)
     
     plt.figure()
     plt.axis('off')
     plt.imshow(image_subtracted, cmap='gray')
     #plt.colorbar()
     #plt.title("g-i: "+images[i], fontsize=10)
     
     # , vmin=vmin,vmax=vmax
     plt.savefig('/home/dobby/LensingProject/training_set/best/' + str(images_g[i].replace("_g.fits",""))+'_g-i.png', bbox_inches='tight') # , dpi=300
   