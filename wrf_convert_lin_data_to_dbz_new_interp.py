#!/usr/bin/python
#from numpy import *
import Nio
#import time
import dircache
import pylab as pl
import dbzcalc_lin_py
import array as array
import numpy as n
import write_zlib_files
import scipy as s
import os
#import scipy.interpolate as sinterp

def compute_tk(tk, pressure, theta,nx,ny,nz):
    p1000mb=100000.
    r_d=287.
    cp=7.*r_d/2.
    for k in range(nz):
	for j in range(ny):
	    for i in range(nx):
		pi=(pressure[i,j,k]/p1000mb)**(r_d/cp)
		tk[i,j,k] = pi*theta[i,j,k]


    return tk





###########################################################################
#########################Reading a file####################################
###########################################################################



#input_directory='/home/scaine/data/wrf_simulations/20060205-lin-d05/d04/'
input_directory=dircache.os.getcwd()+'/'
#input_directory='/media/my_book1/test/d05/'

list_files=dircache.listdir(input_directory)

in0r=0
in0s=0
in0g=0
iliqskin=1
loop=0
infowrite=0
for file in range(len(list_files)):
#for file in range(5,6):


	input_filename=list_files[file]
	if (input_filename[-3:] != '.nc'):
	    continue
	file_in = Nio.open_file(input_directory+input_filename , 'r')
	
	
	file_in_keys=file_in.dimensions.keys()
	file_in_vals=file_in.dimensions.values()
	file_in_attributes=file_in.attributes
	
	##############################Times#############################################
	
	Times = file_in.variables['Times']
	Times_attributes=Times.attributes
	
	##############################XLAT#############################################
	
	XLAT = file_in.variables['XLAT']
	XLAT_attributes=XLAT.attributes
	
	##############################XLONG#############################################
	
	
	XLONG = file_in.variables['XLONG']
	XLONG_attributes=XLONG.attributes
	
	##############################P############################################
	
	P = file_in.variables['P']
	P_attributes = P.attributes
	
	##############################PB############################################
	
	PB = file_in.variables['PB']
	PB_attributes = PB.attributes
	
	############################CREATE PRESSURE################################

	PRES=pl.zeros(pl.shape(P),"float")
	PRES[:,:,:,:] = P[:,:,:,:] + PB[:,:,:,:]


	#############################QRA############################################
	
	QRA = file_in.variables['QRAIN']
	QRA_attributes = QRA.attributes


	#############################QSNOW##########################################
	
	QSN = file_in.variables['QSNOW']
	QSN_attributes = QSN.attributes

	#############################QGRAUP#########################################
	
	QGR = file_in.variables['QGRAUP']
	QGR_attributes = QGR.attributes

	#############################TEMP###########################################
	
	T = file_in.variables['T']
	T_attributes = T.attributes


	#############################QVP###########################################
	
	QVP = file_in.variables['QVAPOR']
	QVP_attributes = QVP.attributes

	###############################Z###########################################
	
	PH = file_in.variables['PH']
	PH_attributes = PH.attributes
	PHB = file_in.variables['PHB']
	PHB_attributes = PHB.attributes

	Z=(PH[:,:,:,:]+PHB[:,:,:,:])/9.81

        ###########################################################################
	##################set up some stuff on the first loop######################
        ###########################################################################

	if (loop ==0):
	    dims=pl.shape(P)
	    ntimes=dims[0]
	    miy=dims[2]
	    mjx=dims[3]
	    mkzh=dims[1]

	    dx=file_in_attributes['DX'][0]	
	    onedegree=111.2*1000.
	    dxfrac=dx/onedegree
	    dxfrac=dxfrac*0.9
	    tox=-1
	    toy=-1
	    fromx=-1
	    fromy=-1
	    for i in range(miy):
		if (XLAT[0,i,0] < (-13.604782104492188+dxfrac)) & (XLAT[0,i,0] > (-13.604782104492188 -dxfrac)):
		    fromx=i
		if (XLAT[0,i,0] < (-10.875259399414062+dxfrac)) & (XLAT[0,i,0] > (-10.875259399414062-dxfrac)):
		    tox=i+1

	    for j in range(mjx):
		if (XLONG[0,0,j] < (129.65934753417969+dxfrac)) & (XLONG[0,0,j] > (129.65934753417969-dxfrac)):
		    fromy=j
		if (XLONG[0,0,j] < (132.45265197753906+dxfrac)) & (XLONG[0,0,j] > (132.45265197753906-dxfrac)):
		    toy=j+1

	    if (tox == -1):
		print 'domain not quite big enough'
		tox=miy
	    
	    if (toy == -1):
		print 'domain notquite big enough'
		toy=mjx

	    if (fromx == -1):
		print 'domain not quite big enough'
		fromx=0

	    if (fromy == -1):
		print 'domain not quite big enough'
		fromy=0
	    xsubset=tox-fromx
	    ysubset=toy-fromy

	    #######check to see if the domain iss the right size#####
	    for checkdom in range(10):
		if ((xsubset*dx)/1000. > 302.5):
		    tox=tox-1
		    xsubset=tox-fromx
		if ((ysubset*dx)/1000. > 302.5):
		    toy=toy-1
		    ysubset=toy-fromy




	    if (xsubset != ysubset):
		print 'WARNING DOMAIN NOT SQUARE'
		print 'chainging domain size'
		if (xsubset > ysubset):
		    tox=tox-1
		    xsubset=tox-fromx

		if (ysubset > xsubset):
		    toy=toy-1
		    ysubset=toy-fromy
	    print 'xsubset'
	    print xsubset
	    print 'ysubset'
	    print ysubset



	    loop=1
	print 'file loaded'


	

	miy=xsubset
	mjx=ysubset

	dbz	=n.arange(miy*mjx*mkzh,dtype="float").reshape(miy,mjx,mkzh)
	qvp	=pl.zeros((xsubset,ysubset,mkzh),"float")
	qra	=pl.zeros((xsubset,ysubset,mkzh),"float")
	qsn	=pl.zeros((xsubset,ysubset,mkzh),"float")
	qgr	=pl.zeros((xsubset,ysubset,mkzh),"float")
	t	=pl.zeros((xsubset,ysubset,mkzh),"float")
	pres	=pl.zeros((xsubset,ysubset,mkzh),"float")
	prs	=pl.zeros((xsubset,ysubset,mkzh),"float")
	tk	=pl.zeros((xsubset,ysubset,mkzh),"float")
	z	=pl.zeros((xsubset,ysubset,mkzh),"float")

        dims=pl.shape(P)
        ntimes=dims[0]

	for time in range(ntimes): 
	    for k in range(mkzh):
		qvp[:,:,k]	=QVP[time,k,fromx:tox,fromy:toy]
		qra[:,:,k]	=QRA[time,k,fromx:tox,fromy:toy]
		qsn[:,:,k]	=QSN[time,k,fromx:tox,fromy:toy]
		qgr[:,:,k]	=QGR[time,k,fromx:tox,fromy:toy]
		t[:,:,k]	=T[time,k,fromx:tox,fromy:toy]+300.
		z[:,:,k]	=Z[time,k,fromx:tox,fromy:toy]
		pres[:,:,k]	=PRES[time,k,fromx:tox,fromy:toy]




	    


	    tk=compute_tk(tk,pres,t,miy,mjx,mkzh)
	    prs[:,:,:]=pres[:,:,:]/100.
	    dbz=dbzcalc_lin_py.dbzcalc(qvp,qra,qsn,qgr,tk,prs,dbz,in0r,in0s,in0g,iliqskin,miy=miy,mjx=mjx,mkzh=mkzh)


	    dbz_interp	=pl.zeros((xsubset,ysubset,40),"float")
	    new_levels  =pl.zeros((40),"float")
	    for level in range(40):
		new_levels[level]=500.*(level+1)

	    for i in range(pl.shape(dbz)[0]):
		for j in range(pl.shape(dbz)[1]):
		    dbz_interp[i,j,:]=s.interp(new_levels[:],z[i,j,:],dbz[i,j,:])
	#	    interp_function=sinterp.interp1d(z[i,j,:],dbz[i,j,:],kind='cubic',bounds_error=False,fill_value=-99.9)
	#	    dbz_interp[i,j,:]=interp_function(new_levels[:])



	    dbz_out	=pl.zeros((40,xsubset*ysubset),"float")
	    for level in range(40):
		dbz_out[level,:]=pl.reshape(dbz_interp[:,:,level],(xsubset*ysubset))



	    outname=''
	    for kk in range(len(Times[time])):
		outname=outname+Times[time][kk]


	    file2write=write_zlib_files.zip_data(dbz_out)


	    dirlist=os.listdir(input_directory)
	    direxist=0
	    for i in range(len(dirlist)):
		if (dirlist[i]=='alllevels'):
		    direxist=1
	    if (direxist==0):
		mkdircommand='mkdir ' + input_directory + '/alllevels'
		os.system(mkdircommand)


	    output_directory=input_directory+'/alllevels/'

	    filewrite=open(output_directory+outname+'alllevels_zlib.ascii','wb')
	    filewrite.write(file2write)
	    filewrite.close
#	    pl.save(output_directory+outname+'alllevels.ascii',dbz_out,fmt='%10.8f')

	    if (infowrite ==0):   
		print 'writing domain info'
		infofile=open(input_directory+'/alllevels/domain_info.txt','wb')
		infofile.write('xsubset')
		infofile.write('\n')
		infofile.write(str(xsubset))
		infofile.write('\n')
		infofile.write('ysubset')
		infofile.write('\n')
		infofile.write(str(ysubset))
		infofile.write('\n')
		infofile.write('gridspacing')
		infofile.write('\n')
		infofile.write(str(dx))
		infofile.write('\n')
		infofile.close()
		infowrite =1



	file_in.close()


