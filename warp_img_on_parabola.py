import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cv2
import pdb



def Rx(theta):
	theta = np.deg2rad(theta)
	return np.array([[1.,0,0],[0,np.cos(theta),np.sin(theta)],[0,-np.sin(theta),np.cos(theta)]])

def Ry(theta):
	theta = np.deg2rad(theta)
	return np.array([[np.cos(theta),0,-np.sin(theta)],[0,1.,0],[np.sin(theta),0,np.cos(theta)]])

def Rz(theta):
	theta = np.deg2rad(theta)
	return np.array([[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1.]])

def traverse_prarabola_len(xi,a,li,dl):
	# Arc Length of a parabola
	arc_len = lambda x: 0.25 * (x*np.sqrt(1 + x**2) + np.log(x + np.sqrt(1 + x**2)))/a
	fun = lambda l: np.abs(li + dl - arc_len(2*a*l))
	ret = minimize_scalar(fun,bounds=(xi,xi+dl),tol=1e-3)
	return ret.x

def map_pix2parabola(a, nsteps):
	f = lambda x: a * x**2 

	xi,li,dl = 0.0,0.0,1.0
	x = [0.0]
	y = [0.0]
	for i in range(nsteps):
		xj = traverse_prarabola_len(xi,a,li,dl)
		yj = f(xj)
		xi,li = xj,li+dl
		x = [-1*xj] + x + [xj]
		y = [yj] + y + [yj]

	return np.array(x), np.array(y)


def warp_img(img,alpha=0.01,dims=(257,257),rvec=(0.0,0.0,0.0)):
	alpha = abs(alpha) + 1e-15 # Pathological input
	if(len(img.shape) ==2 ):
		img = np.dstack((img,img,img))
	
	h,w,d = dims[0], dims[1], 3
	img = cv2.resize(img,(h,w))

	K = np.array([[w//2,0,w//2],[0,h//2,h//2], [0,0,1]]) # Camera Interinsic parameters

	hc,wc = h//2, w//2

	# Map pixes of each row to a parabola
	# Prabola defines the depth
	xp, zp = map_pix2parabola(alpha, wc)

	xv, yv = np.meshgrid(xp,np.linspace(-hc,hc,h))
	zv = h//2 + zp[np.newaxis,:].repeat(h,axis=0)

	I = img.reshape(-1,img.shape[-1])
	X = np.vstack((xv.reshape((1,h*w)), yv.reshape((1,h*w)), zv.reshape((1,h*w))))

	Xmu = X.mean(axis=1)[:,np.newaxis]
	R = Rx(rvec[0]).dot(Ry(rvec[1])).dot(Rz(rvec[2]))
	X = R.dot(X-Xmu) + Xmu

	# Project to Image plane
	Y = K.dot(X)
	Y = Y/Y[-1,:]

	xt, yt = np.meshgrid(np.linspace(0,w-1,w),np.linspace(0,h-1,h))

	out_img = griddata((Y[0,:],Y[1,:]),I.astype(float),
					  (xt.reshape((1,h*w)),yt.reshape((1,h*w))),
					  method='linear',fill_value=0.0)

	return out_img.reshape((h,w,d)).astype(np.uint8)


if __name__== "__main__":

	img = cv2.imread('check.jpg') # Load an image
	img = img[:,:,::-1] # BGR to RGB convert
	plt.figure(1)
	plt.imshow(img)
	# plt.title('Original Image')

	out_img = warp_img(img,alpha=0.0)
	plt.figure(2)
	plt.subplot(121)
	plt.imshow(out_img)
	plt.xlabel(r'$\alpha=0.0$')
	
	out_img = warp_img(img,alpha=0.005)
	plt.subplot(122)
	plt.imshow(out_img)
	plt.xlabel(r'$\alpha=0.005$')
	# plt.suptitle('Warped Images',fontsize=16)

	out_img_rx = warp_img(img,alpha=0.005,rvec=(20.0,0.0,0.0))
	out_img_ry = warp_img(img,alpha=0.005,rvec=(0.0,20.0,0.0))
	out_img_rz = warp_img(img,alpha=0.005,rvec=(0.0,0.0,20.0))
	plt.figure(3)
	# plt.suptitle('Warping and 3D Roatation',fontsize=16)
	plt.subplot(131)
	plt.imshow(out_img_rx)
	plt.title('Rotation in ZY-Plane')
	plt.xlabel(r'$\alpha=0.005,\theta_x=20deg$')
	plt.subplot(132)
	plt.imshow(out_img_ry)
	plt.title('Rotation in ZX-Plane')
	plt.xlabel(r'$\alpha=0.005,\theta_y=20deg$')
	plt.subplot(133)
	plt.imshow(out_img_rz)
	plt.title('Rotation in XY-Plane')
	plt.xlabel(r'$\alpha=0.005,\theta_z=20deg$')
	plt.show()

