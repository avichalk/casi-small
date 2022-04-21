from astropy.io import fits
#import matplotlib.pyplot as plt
import numpy as np
import json
import os


def main():
	#params = json.load(open("hypers_3.json", "r"))
	#params.["data"]
	data_path = "/groups/yshirley/cnntrain/yes"
	for item in os.listdir(data_path):
		mask_image(item, fits.open(f"{data_path}/{item}"))
	
def mask_image(image_name, image, threshold=25):
	data = image[0].data
	mask = data > threshold
	hdu = fits.PrimaryHDU(np.asarray(mask))
	#hdul = fits.HDUList([hdu])
	#hdul.writeto(f"~/mask/{image_name}_mask.fits")

	#hdu.writeto(f"~/mask/{image_name}_mask.fits")
	#plt.imshow(mask)
	#plt.savefig(f"~/mask/{image_name}_mask.fits")

main()
	
