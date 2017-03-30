This is assignment3 for CSCI3190 Computational Photography in CUHK.
It generates an HDR photo by using a serie of photos with different exposure time to reconstruct a radiance map and apply tone mapping.


Requirements 

Basic part
1. Radiance Map
	1.1 Sampling method: the sampling pixels are of equal distance with each other
	1.2 Construct the matrix
	1.3 Calculate the image radiance (Ei)
		Use weight index to calcualte lnEi
	# functions: np.zeros(), np.append(), scipy.sparse.coo_matrix(), scipy.sparse.linalg.lsqr()

2. Global Tone Mapping
	Use L/L+1 or L/L+2 based on image effects

3. Tone Mapping
	Get result from power 0.2 of the chrominance based on image effects
	# functions: np.nanmax(), np.nanmin(), cv2.bilateralFilter(), np.power(), np.log2(), np.subtract()
	
4. Bilateral Filter
	Compute the filtered image with convolutions 
	HDR images are created but the speed is slower than openCV function 
	# functions: np.exp(), np.power(), np.abs()
