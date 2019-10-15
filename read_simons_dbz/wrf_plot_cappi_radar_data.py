#!/usr/bin/python
import matplotlib
matplotlib.use('agg')
import pylab as pl
import numpy as n
import time as t
import dircache
from sys import argv
import os
import read_zlib_files


directory=dircache.os.getcwd()+'/'

listfiles=dircache.listdir(directory)

domain_info=[]
fid=open(directory+'/domain_info.txt')
for line in fid:
    domain_info.append(line.split()[0])
fid.close()
xgridpoints=int(domain_info[1])
ygridpoints=int(domain_info[3])
delx=float(domain_info[5])/1000.			#must convert to km
dely=delx
print 'xgridpoints'
print xgridpoints
print 'xgridpoinis'
print ygridpoints
print 'delxy'
print delx


dbzdata=pl.zeros((40,xgridpoints,ygridpoints),"float")

for time in range(len(listfiles)):
    pl.clf()

    filename= listfiles[time]
    if (filename=='mdv_out'):
	continue
    if (filename=='domain_info.txt'):
	continue



    if (filename[-10:] == 'zlib.ascii'):
	doave=1 	# this is not used, fix later
	data=read_zlib_files.read_wrf(directory+filename,doave,xgridpoints,ygridpoints)
	for level in range(40):
	    dbzdata[level,:,:]=pl.reshape(data[level,:],(xgridpoints,ygridpoints))
    else:
	continue


    xrange=300
    yrange=300

    xmina=(-xrange/2)
    ymina=(-yrange/2)


    dar_coast=n.loadtxt('dar_coast.dat')

    dbz_plane=pl.zeros((xgridpoints,ygridpoints),"float")

    #change dbzdata[number,:,:] do get different levels

    #level 4=2.5km = origional	
    #level 9=5.0km 
    #level 14=7.5km
    #level 19=10.0 km
    #level 24=12.5 km
    #level 29=15. km
    
    dbz_plane[:,:]=dbzdata[9,:,:]

    xar=pl.zeros(xgridpoints,"float")
    yar=pl.zeros(ygridpoints,"float")
    for ii in range(xgridpoints):
        xar[ii]=xmina+(ii)*delx
    for jj in range(ygridpoints):
        yar[jj]=ymina+(jj)*dely

    dbz_mask=n.ma.masked_less(dbz_plane,-10)

    dar_coast_x=(dar_coast[:,0]-131.0444)*111.12*n.cos((n.pi/180)*dar_coast[:,1])
    dar_coast_y=(dar_coast[:,1]+12.24917)*111.12

    cma=pl.get_cmap('jet',lut=10)
    pl.plot(dar_coast_x,dar_coast_y,'k',linewidth=2)
    if (dbz_mask.count() == 0):
	dbz_mask[0,0]=0.0
	try:
           cappi=pl.pcolor(xar-delx/2,yar-dely/2,dbz_mask[:,:],cmap=cma)
	except:
           cappi=pl.pcolor(dbz_mask[:,:],cmap=cma)
	cappi.set_clim((10,60))


    if (dbz_mask.count() >0):
	dbz_mask[0,0]=0.0
	try:
           cappi=pl.pcolor(xar-delx/2,yar-dely/2,dbz_mask[:,:],cmap=cma)
	except:
           cappi=pl.pcolor(dbz_mask[:,:],cmap=cma)
        cappi.set_clim((10,60))
        pl.colorbar()

    pl.axis([-150,150,-150,150],'equal')

    if (filename[-10:] == 'zlib.ascii'):
	datename=filename[:-20]
    else:
	datename=filename[:-15]
    print datename
    pl.title(datename,fontsize=25)

    dirlist=os.listdir(directory)
    direxist=0
    for i in range(len(dirlist)):
	if (dirlist[i]=='plots'):
	    direxist=1
    if (direxist==0):
	mkdircommand='mkdir ' + directory + '/plots'
	os.system(mkdircommand)


    outfilename=directory+'/plots/'+datename+'-dbz.png'
    pl.savefig(outfilename)
