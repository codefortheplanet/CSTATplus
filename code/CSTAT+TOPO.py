#CSTAT+ A GPU-accelerated spatial pattern analysis algorithm for high-resolution 2D/3D hydrologic connectivity using array vectorization and convolutional neural network 
#Author: Feng Yu, Jonathan M. Harbor
#Department of Earth, Atmospheric and Planetary Sciences, Purdue University, 550 Stadium Mall Dr, West Lafayette, IN 47907 USA.
#Email: yu172@purdue.edu; Alternative: fyu18@outlook.com
#This is the directional version CSTAT+/TOPO

import numpy as np
import heapq as pq
import os
from itertools import chain
import copy as cp
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_dilation
import math as ma
from osgeo import gdal 
from scipy.ndimage.measurements import label
from itertools import combinations,combinations_with_replacement,product
from mxnet import nd,gpu          
from timeit import default_timer as timer
import pandas as pd

#Process DEM and amend flow directions for multiple flow direction method
def demprocess(dem_ori,NoData,rows,cols,ctx):
    dem_fill=priorityqueue(rows,cols,dem_ori)
    edgecells,pourpoints=depressions(dem_fill,dem_ori)
    switch=1
    flatdep,flowdir,d_flat=flowdr(dem_fill,NoData,rows,cols,ctx,switch)
    flatregion=np.asarray(np.where(flatdep>=-3),dtype="int32").T#All flat regions
    flatregion_exlow=np.asarray(np.where((flatdep==-2)+(flatdep==-3)),dtype="int32").T
    highedges=np.asarray(np.where(flatdep==-3),dtype="int32").T
    lowedges=np.asarray(np.where(flatdep==-1),dtype="int32").T
    kernel_dict=dictionary(1)
    if lowedges.shape[0]>0:
        flatlabel=findregions(flatregion,rows,cols,kernel_dict,d_flat)#Label flats into adjacent regions of same elevation
        flatmask,flatheight,kernel_awayfrom=awayfromhigher(highedges,kernel_dict,rows,cols,flatlabel,flatregion_exlow,flowdir,ctx)
        flatmask=awayfromlower(lowedges,kernel_dict,rows,cols,flatlabel,flatregion,flatmask,flatheight,kernel_awayfrom,flowdir)
        switch=3
        flowdir_flat=flowdr(flatmask,NoData,rows,cols,ctx,switch)
        flowdir_amend=((flowdir!=-1)*(flowdir!=-999))*flowdir+((flowdir==-1)+(flowdir==-999))*flowdir_flat
        return (flowdir_amend,edgecells,pourpoints,kernel_dict) 
    else:
        return (flowdir,edgecells,pourpoints,kernel_dict)

def priorityqueue(rows,cols,dem_ori):       
    Closed=np.zeros((rows,cols),dtype="int32")    
    dem_fill=cp.deepcopy(dem_ori)
    #Create coordinates at edges
    width=list(range(cols))
    height=list(range(1,rows-1))
    zero_w,zero_h=[0 for i in range (cols)],[0 for i in range (rows-2)]
    max_w,max_h=[cols-1 for i in range (cols)],[rows-1 for i in range (rows)]
    firstrow=list(zip(zero_w,width))
    lastrow=list(zip(max_h,width))
    firstcol=list(zip(height,zero_h))
    lastcol=list(zip(height,max_w))
    edgecoor=list(chain(firstrow,lastrow,firstcol,lastcol))

    #Priority queue
    Open=[]
    pq.heapify(Open)

    #Plain queue (not prioritized)
    Pit=[]

    for i in edgecoor:
        Closed[i]=1
        element=[dem_fill[i],i]
        Open.append(element)
        
    #Generate kernel structure starting from E and counterclockwise
    k=[(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1)]

    #while either Open or Pit is not empty do
    #if Pit is not empty then
    #c←pop(Pit)
    #else
    #c←pop(Open)
    while len(Open)>0 or len(Pit)>0:
        if len(Pit)>0:
            c=Pit.pop(0)
        else:
            c=pq.heappop(Open)

    #for all neighbors n of c do
    #if Closed(n) then repeat loop
    #Closed(n)←TRUE

        for i in k:
            nx=c[1][0]+i[0]
            ny=c[1][1]+i[1]
            if (nx>=0)*(nx<=rows-1)*(ny>=0)*(ny<=cols-1)==1:
                if Closed[nx,ny]==0:
                    Closed[nx,ny]=1
                    if dem_fill[nx,ny]<=c[0] and dem_fill[nx,ny]!=NoData:
                        if dem_fill[nx,ny]<c[0]:
                            dem_fill[nx,ny]=c[0]
                        Pit.append([dem_fill[nx,ny],(nx,ny)])
                    else:
                        Open.append([dem_fill[nx,ny],(nx,ny)])
    return (dem_fill)

def depressions(dem,dem_ori):
    #Locate edge cells and pour points for each depression catchment
    internal=dem-dem_ori
    connection_structure = np.array([[1,1,1],[1,1,1],[1,1,1]])
    dpregions_internal, num_features =label (internal,structure=connection_structure)
    dpregions_full=binary_dilation(internal,structure=connection_structure)*1
    dpregions_full, num_features =label (dpregions_full,structure=connection_structure)
    dpregions_edge=dpregions_full-dpregions_internal

    min_edge_coor=[]
    coor_edge=[]

    for i in range (1,np.amax(dpregions_edge)+1):
        coor_edge_region=np.asarray(np.where(dpregions_edge==i),dtype="int32").T
        z_edge=[]
        min_edge_coor_region=[]
        for j in range (coor_edge_region.shape[0]):
            z_edge.append(dem_ori[coor_edge_region[j,0],coor_edge_region[j,1]])
        z_edge=np.array(z_edge)
        min_edge=np.amin(z_edge)
        min_edge_coor_1D=np.asarray(np.where(z_edge==min_edge),dtype="int32")[0]
        min_edge_coor_region=coor_edge_region[min_edge_coor_1D]
        min_edge_coor.append(min_edge_coor_region)
        coor_edge.append(coor_edge_region)
    return (coor_edge,min_edge_coor)

def flowdr(dem_fill,NoData,rows,cols,ctx,switch):
    ingrid = np.indices((rows, cols))
    ingrid[0]        # row indices
    ingrid[1]        # column indices
    ingridxmx=nd.array(ingrid[1],ctx[0]).reshape((1,1,rows, cols))
    ingridymx=nd.array(ingrid[0],ctx[0]).reshape((1,1,rows, cols))
    dem_fillmx=nd.array(dem_fill,ctx[0])
    demmx=dem_fillmx.reshape((1,1,rows, cols))
    res=1
    l=[0,1,2,3,4,5,6,7,0]
    direct=[1,2,4,8,16,32,64,128]
    direct_d=[[1,3],[2,6],[4,12],[8,24],[16,48],[32,96],[64,192],[128,129]]
    weight=[None]*8
    weight1=[None]*8
    convx=[None]*8
    convy=[None]*8
    convz=[None]*8
    runlen=[1,ma.pow(2,0.5),1,ma.pow(2,0.5),1,ma.pow(2,0.5),1,ma.pow(2,0.5)]*res
    n = [[[] for x in range(3)] for x in range(8)]#create list to store normal vectors for each facet
    s = [None]*8
    d = [None]*8

    weight[0] = nd.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]], gpu(0))
    weight[1] = nd.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]], gpu(0))
    weight[2] = nd.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]], gpu(0))
    weight[3] = nd.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]], gpu(0))
    weight[4] = nd.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], gpu(0))
    weight[5] = nd.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0]], gpu(0))
    weight[6] = nd.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]], gpu(0))
    weight[7] = nd.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]], gpu(0))
    
    weight1[0] = nd.array([[0, 0, 0], [0, 1, -10], [0, 0, 0]], gpu(0))
    weight1[1] = nd.array([[0, 0, -10], [0, 1, 0], [0, 0, 0]], gpu(0))
    weight1[2] = nd.array([[0, -10, 0], [0, 1, 0], [0, 0, 0]], gpu(0))
    weight1[3] = nd.array([[-10, 0, 0], [0, 1, 0], [0, 0, 0]], gpu(0))
    weight1[4] = nd.array([[0, 0, 0], [-10, 1, 0], [0, 0, 0]], gpu(0))
    weight1[5] = nd.array([[0, 0, 0], [0, 1, 0], [-10, 0, 0]], gpu(0))
    weight1[6] = nd.array([[0, 0, 0], [0, 1, 0], [0, -10, 0]], gpu(0))
    weight1[7] = nd.array([[0, 0, 0], [0, 1, 0], [0, 0, -10]], gpu(0))

    d0=nd.zeros((rows, cols),ctx[0],dtype='float32')
    dd=nd.zeros((rows, cols),ctx[0],dtype='float32')
    d_flat=nd.zeros((rows, cols),ctx[0],dtype='float32')
    flat=nd.zeros((rows, cols),ctx[0],dtype='float32')
    dep=nd.zeros((rows, cols),ctx[0],dtype='float32')
    high=nd.zeros((rows, cols),ctx[0],dtype='float32')
    fd=nd.zeros((rows, cols),ctx[0],dtype='float32')-999
    d_compact=nd.zeros((rows, cols),ctx[0],dtype='float32')-1

    for i in range(0,8):
        w=weight[i].reshape((1, 1, 3, 3))
        convz[i] = nd.Convolution(data=demmx, weight=w, kernel=(3,3), no_bias=True, num_filter=1,pad=(1,1),cudnn_tune='off')
        convz[i]=convz[i][0,0,:,:]
        if switch==1 or 3:
            convx[i] = nd.Convolution(data=ingridxmx, weight=w, kernel=(3,3), no_bias=True, num_filter=1,pad=(1,1),cudnn_tune='off')
            convy[i] = nd.Convolution(data=ingridymx, weight=w, kernel=(3,3), no_bias=True, num_filter=1,pad=(1,1),cudnn_tune='off')        
            convx[i]=convx[i][0,0,:,:]
            convy[i]=convy[i][0,0,:,:]
        
    if switch==1 or 3:
        for p in range(0,8):#8 facets from N-NE clockwise
            l0=l[p]
            l1=l[p+1]
            d[l0]=d0-999#Nodata value
            dmax=d0-999
            smax=d0-999
            n[l0][0]= convz[l0]*convy[l1]-convz[l1]*convy[l0]#nx
            n[l0][1]= convz[l0]*convx[l1]-convz[l1]*convx[l0]#ny
            n[l0][2]= convy[l0]*convx[l1]-convy[l1]*convx[l0]#nz
            #make boolean mask to determine direction d and slope s
            d[l0]=nd.where(condition=((n[l0][0]==0)*(n[l0][1]>=0)),x=d0,y=d[l0])

            d[l0]=nd.where(condition=((n[l0][0]==0)*(n[l0][1])<0),x=d0+ma.pi,y=d[l0])

            d[l0]=nd.where(condition=(n[l0][0]>0),x=ma.pi/2-nd.arctan(n[l0][1]/n[l0][0]),y=d[l0])

            d[l0]=nd.where(condition=(n[l0][0]<0),x=3*ma.pi/2-nd.arctan(n[l0][1]/n[l0][0]),y=d[l0])


            d[l0]=nd.where(condition=((convz[l0]<=0)*(convz[l1]<=0)),x=dmax,y=d[l0])

            s[l0]=-nd.tan(nd.arccos(n[l0][2]/(nd.sqrt(nd.square(n[l0][0])+nd.square(n[l0][1])+nd.square(n[l0][2])))))#slope of the triangular facet
            s[l0]=nd.where(condition=((convz[l0]<=0)*(convz[l1]<=0)),x=smax,y=s[l0])
            #Modify the scenario when the steepest slope is outside the 45 range of each facet
            dmax=nd.where(condition=((convz[l0]/runlen[l0]>=convz[l1]/runlen[l0])*(convz[l0]>0)),x=d0+ma.pi*l0/4,y=dmax)
            dmax=nd.where(condition=((convz[l0]/runlen[l0]<convz[l1]/runlen[l0])*(convz[l1]>0)),x=d0+ma.pi*(l0+1)/4,y=dmax)

            smax=nd.where(condition=((convz[l0]>=convz[l1])*(convz[l0]>0)),x=convz[l0]/runlen[l0],y=smax)
            smax=nd.where(condition=((convz[l0]<convz[l1])*(convz[l1]>0)),x=convz[l1]/runlen[l1],y=smax)
            d[l0]=nd.where(condition=((d[l0]<ma.pi*l0/4)+(d[l0]>ma.pi*l1/4)),x=dmax,y=d[l0])

            s[l0]=nd.where(condition=((d[l0]<ma.pi*l0/4)+(d[l0]>ma.pi*l1/4)),x=smax,y=s[l0])

            if switch==1:

                #flat and depressions indicator grid    

                flat=(convz[l0]==0)+flat
                dep=(convz[l0]<0)+dep
                high=(convz[l0]>0)+high

        for q in range(0,8):#check if the 45 degree range angles need to be maintaied, otherwise delete (set to NoData)
            l0=l[q]
            l1=l[q+1]
            l2=l[q-1]
            dmax=d0-999
            if q==0:
                dmax=nd.where(condition=(d[0]==d[1]),x=d[0],y=dmax)
                dmax=nd.where(condition=(d[0]==d[7]),x=d[0],y=dmax)
                d[0]=nd.where(condition=((d[0]==ma.pi*l0/4)+(d[0]==ma.pi*l1/4)),x=dmax,y=d[0])
            else:
                dmax=nd.where(condition=(d[l0]==d[l1]),x=d[l0],y=dmax)
                dmax=nd.where(condition=(d[l0]==d[l2]),x=d[l0],y=dmax)
                d[l0]=nd.where(condition=((d[l0]==ma.pi*l0/4)+(d[l0]==ma.pi*l1/4)),x=dmax,y=d[l0])
    #Check if flat or surface depression area. then lable with -1 or -10 respectively

    if switch==1:

        fd=nd.where(condition=(flat==8),x=d0-2,y=fd)#flats

        fd=nd.where(condition=(dep>=1)*(high==0),x=d0-3,y=fd)#high edge

        high_zero=nd.where(condition=(high==0),x=d0+1,y=d0)
    
    
    for j in range (0,8):
        if switch==1 or switch==2:
            d_flat=nd.where(condition=(convz[j]==0),x=d0+direct[j],y=d0)+d_flat
        
        if switch==1:
            flat_near=nd.where(condition=(convz[j]==0),x=d0+5,y=d0)
            dd1=high_zero+flat_near
            w=weight1[j].reshape((1, 1, 3, 3))
            dd1=dd1.reshape((1,1,rows, cols))
            conv_near= nd.Convolution(data=dd1, weight=w, kernel=(3,3), no_bias=True, num_filter=1,pad=(1,1),cudnn_tune='off')
            conv_near= conv_near[0,0,:,:]
            dd=nd.where(condition=(conv_near==-5)+(conv_near==-59)+(conv_near==-54)+(conv_near==-4),x=d0+1,y=d0)+dd

        if switch==1 or switch==3:
            d_compact=nd.where(condition=(d[j]==ma.pi*j/4),x=d0+direct_d[j][0],y=d_compact)
            d_compact=nd.where(condition=(d[j]>j*ma.pi/4)*(d[j]<(j+1)*ma.pi/4),x=d0+direct_d[j][1],y=d_compact)

    if switch==1 or switch==3:
        d_compact=nd.where(condition=(dem_fillmx==d0+NoData),x=d0-999,y=d_compact)#NoData        
    
    if switch==1:
        fd=nd.where(condition=(dd>=1)*(high>=1),x=d0-1,y=fd)#low edge
        fd=nd.where(condition=(dep==8),x=d0-10,y=fd)#lowest points in depressions
        return (fd.asnumpy(),d_compact.asnumpy(),d_flat.asnumpy())

    if switch==2:
        return (d_flat.asnumpy())
    if switch==3:
        return (d_compact.asnumpy())

def dictionary(dictype):#Make dictionary to determine the 8 cardinal directions for each cell
    
    kernel_label=[1,2,4,8,16,32,64,128]
    kernel_shift=[(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1)]
    kl=[]
    ks=[]
    kl_sum=[]
    for i in range(1,9):
        kl+=list(combinations(kernel_label,i))
        ks+=list(combinations(kernel_shift,i))
    for i in kl:
        kl_sum.append(sum(i))
    kernel_dict=dict(zip(kl_sum, ks))
    return (kernel_dict)

def findregions(region,rows,cols,kernel_dict,di):
    label=np.zeros((rows,cols),dtype="int32")
    region_num=0
    s = set(tuple(map(tuple, region)))
    marker=[]
    for i in region:
        if tuple(i) not in marker:
            region_num+=1
            stack=[tuple(i)]
            label[tuple(i)]=region_num 
            while stack:
                x,y=stack.pop(0)
                kernel=kernel_dict.get(di[x,y])
                if (x<=label.shape[0]-1 and x>= 0) * (y<=label.shape[1]-1 and y>=0) * (kernel!=None):
                        for i in kernel:
                            xx=x+i[0]
                            yy=y+i[1]
                            if (xx,yy) not in marker and (xx,yy) in s:
                                label[xx,yy]=region_num
                                stack.append((xx,yy))
                                marker.append((xx,yy))
    return(label)

def awayfromhigher(highedges,kernel_dict,rows,cols,flatlabel,region_excludelow,flowdir,ctx):
    flatmask=np.zeros((rows,cols),dtype="int32")
    flatheight=[None]*np.amax(flatlabel)
    loops=1
    switch=2
    kernel_awayfrom=flowdr(flatlabel,NoData,rows,cols,ctx,switch)
    l=list(tuple(map(tuple, highedges)))
    s = set(tuple(map(tuple, region_excludelow)))
    endmarker=(-1,-1)
    l.append(endmarker)
    marker=[]
    while len(l)>1:
        c=l.pop(0)
        if c==endmarker:
            loops+=1
            l.append(endmarker)
        else:
            if flatmask[c]==0:
                flatmask[c]=loops
                flatheight[flatlabel[c]-1]=loops
                kernel=kernel_dict.get(kernel_awayfrom[c])
                if kernel!=None:
                    x,y=c
                    for i in kernel:
                        xx=x+i[0]
                        yy=y+i[1] 
                        if (xx,yy) in s:
                            if flatlabel[xx,yy]==flatlabel[x,y] and (xx,yy) not in marker and flowdir[xx,yy]==-1:
                                l.append((xx,yy))
                                marker.append((xx,yy))
    return(flatmask,flatheight,kernel_awayfrom)

def awayfromlower(lowedges,kernel_dict,rows,cols,flatlabel,region,flatmask,flatheight,kernel_awayfrom,flowdir):
    flatmask=flatmask*-1
    loops=1
    l=list(tuple(map(tuple, lowedges)))
    s = set(tuple(map(tuple, region)))
    endmarker=(-1,-1)
    l.append(endmarker)
    marker=[]
    while len(l)>1:
        c=l.pop(0)
        if c==endmarker:
            loops+=1
            l.append(endmarker)
        else:
            if flatmask[c]<0:
                flatmask[c]=2*loops+flatheight[flatlabel[c]-1]+flatmask[c]
                kernel=kernel_dict.get(kernel_awayfrom[c])
                if kernel!=None:
                    x,y=c
                    for i in kernel:
                        xx=x+i[0]
                        yy=y+i[1] 
                        if (xx,yy) in s: 
                            if flatlabel[xx,yy]==flatlabel[x,y] and (xx,yy) not in marker and flowdir[xx,yy]==-1:
                                l.append((xx,yy))
                                marker.append((xx,yy))
            if flatmask[c]==0:
                flatmask[c]=2*loops
                kernel=kernel_dict.get(kernel_awayfrom[c])
                if kernel!=None:
                    x,y=c
                    for i in kernel:
                        xx=x+i[0]
                        yy=y+i[1] 
                        if (xx,yy) in s: 
                            if flatlabel[xx,yy]==flatlabel[x,y] and (xx,yy) not in marker and flowdir[xx,yy]==-1:
                                l.append((xx,yy))
                                marker.append((xx,yy))
    flatmask[flatmask==0]=np.amax(flatmask)+2
    return(flatmask)

#Binarize pattern
def prep(expe,threshold,NoData):
    #Provide threshold for High/Low, usually the depth of shallow sheetflow
    expe1=cp.deepcopy(expe)
    expe2=cp.deepcopy(expe)
    expe1[(expe1>=threshold)]=1
    expe1[(expe1<threshold)]=0
    expe2[(expe2==NoData)]=-1
    expe2[(expe2>0)]=0
    connection_structure = np.array([[1,1,1],[1,1,1],[1,1,1]])
    expela, num_features =label (expe1,structure=connection_structure)
    expe3=expe2+expela
    return (expe3)

def itercontrol(regions,k,bins,dibins,dibins4,binnum,rows,cols,flowdir,kernel_dict,ctx):
    broadcdp1=1000000
    #Initiate empty array for storing histogram for directions, distances, and number of counted pairs in each distance range bin
    co0=np.zeros(binnum-1,dtype="float32")
    codi0=np.zeros((4,binnum-1),dtype="float32")
    count0=np.zeros(binnum-1,dtype="float32")
    count4=np.zeros((4,binnum-1),dtype="float32")
    co4=np.zeros((4,binnum-1),dtype="float32")
    bins=np.array(bins)
    dibins=np.array(dibins)
    dibins4=np.array(dibins4)
    rmrep=np.ones((rows,cols),dtype="int32")
    if k==2:
        for t in regions:
            x,y=t
            rmrep[x,y]=0
            if x>=0 and y>=0 and x<rows and y<cols:
                flowtrack=downslope(x,y,regions,rows,cols,flowdir,kernel_dict)
                flowtrack=flowtrack*rmrep
                flowpin=np.asarray(np.where(flowtrack==1),dtype="int32").T
                if flowpin.shape[0]!=0:
                    #Create segment index for the input array to meet the memory requirement
                    imax=list(range(int(flowpin.shape[0]/broadcdp1)+(flowpin.shape[0]%broadcdp1!=0)))
                    for i in imax:
                        vout=distanceAATOPO(flowpin,i,binnum,dibins,dibins4,x,y,ctx)
                        co0+=vout[0]
                        codi0+=vout[1]
                        count0+=vout[2]
                        co4+=vout[3]
                        count4+=vout[4]
        return (co0,codi0,count0,co4,count4)
    elif k==1:
#Create segment index for the input array to meet the memory requirement
        imax=list(range(int(regions.shape[0]/broadcdp)+(regions.shape[0]%broadcdp!=0)))
#Combinations with repeated indicies
        iterator=list(combinations_with_replacement(imax,2))
        for i in iterator:
            if i[0]==i[1]:
                count0+=distance2(regions,i,binnum,bins,ctx)        
            else:
                count0+=distance1(regions,i,binnum,bins,ctx)
        return (count0)
    else:
#Unpack the tuple
        regions_high,regions_low=regions        
#Create segment index for the input array to meet the memory requirement
        imax_high=list(range(int(regions_high.shape[0]/broadcdp)+(regions_high.shape[0]%broadcdp!=0)))
        imax_low=list(range(int(regions_low.shape[0]/broadcdp)+(regions_low.shape[0]%broadcdp!=0)))
#Combinations with repeated indicies
        iterator=list(product(imax_high,imax_low))
        for i in iterator:
            count0+=distance11(regions_high,regions_low,i,binnum,bins,ctx)
        return (count0)

def downslope(x,y,regions,rows,cols,flowdir,kernel_dict):
    flowtrack=np.zeros((rows,cols),dtype="int32")
    s = set(tuple(map(tuple, regions)))
    marker=[(x,y)]
    stack=[(x,y)]
    flowtrack[x,y]=1
    while stack:
        c=stack.pop(0)
        if flowdir[c]!=-1:
            kernel=kernel_dict.get(flowdir[c])
            x,y=c
            for i in kernel:
                xx=x+i[0]
                yy=y+i[1] 
                if (xx,yy) in s and (xx,yy) not in marker:
                    flowtrack[xx,yy]=1#(xx<=rows-1 and xx>= 0) * (yy<=cols-1 and yy>=0):
                    stack.append((xx,yy))
                    marker.append((xx,yy))
    return(flowtrack)

def distanceAATOPO(regions,i,binnum,dibins,dibins4,x,y,ctx):
#Initiate empty array for storing histogram for directions, distances, and number of counted pairs in each distance range bin
    co0=nd.zeros(binnum-1,ctx[0],dtype="float32")
    codi0=nd.zeros((5,binnum-1),ctx[0],dtype="float32")
    count0=nd.zeros(binnum-1,ctx[0],dtype="float32")
    count4=nd.zeros((5,binnum-1),ctx[0],dtype="float32")
    co4=nd.zeros((5,binnum-1),ctx[0],dtype="float32")
    
#Calculate index coordinates and directions by chuncks
    a=regions[i*broadcdp:min((i+1)*broadcdp,regions.shape[0]),:]
    a1=nd.array(a,ctx[0])
    b1=nd.array([x,y],ctx[0])
    a1_b1=(nd.expand_dims(a1,axis=1)-b1).reshape((-1,2))
    x1_x2=a1_b1[:,0]
    y1_y2=a1_b1[:,1]
#Find the rows where all equal zeros
    boolmask=(x1_x2==0)*(y1_y2==0)
    labels=nd.zeros(boolmask.shape[0],ctx[0],dtype="float32")
    sdi0=(nd.degrees(nd.arctan((y1_y2)/(x1_x2)))+90).reshape((-1,))
    ldis=nd.broadcast_hypot(x1_x2,y1_y2).reshape((-1,))
#Change the zeros into -1
    sdi0=nd.where(condition=boolmask,x=labels-1,y=sdi0)
    ldis=nd.where(condition=boolmask,x=labels-1,y=ldis)
#Change 0 to 180 so it can apply sum of boolean mask without losing values        
    sdi0=nd.where(condition=(sdi0==0),x=labels+180,y=sdi0)
#Store sum of distances co0 and histogram of directions in each range bin
    for p in range (0,binnum-1):
        booleanmask=nd.equal((ldis>=bins[p]),(ldis<bins[p+1]))
        count0[p]+=nd.sum(booleanmask)
        co0[p]+=nd.sum(ldis*booleanmask)
#Exclue values not in distance range bin
        sdi1=nd.where(condition=(booleanmask==0),x=labels-1,y=sdi0)
        for q in range (0,5):
            booleanmaskdi=nd.equal((sdi1>=dibins[q]),(sdi1<dibins[q+1]))            
            codi0[q,p]+=nd.nansum(booleanmaskdi)
            
    for k in range (0,5):
        booleanmaskdi=nd.equal((sdi0>=dibins4[k]),(sdi0<dibins4[k+1]))
        ldis0=ldis*booleanmaskdi
        for l in range (0,binnum-1):
            booleanmask=nd.equal((ldis0>=bins[l]),(ldis0<bins[l+1]))
            count4[k,l]+=nd.sum(booleanmask)
            co4[k,l]+=nd.sum(ldis0*booleanmask)
            
    codi0[0,:]+=codi0[4,:]
    codi0=codi0[0:4,:]
    count4[0,:]+=count4[4,:]
    count4=count4[0:4,:]
    co4[0,:]+=co4[4,:]
    co4=co4[0:4,:]
    return(co0.asnumpy(),codi0.asnumpy(),count0.asnumpy(),co4.asnumpy(),count4.asnumpy())

#Full permutation distance computation
def distance1(regions,i,binnum,bins,ctx):
#Initiate empty array for storing the number of counted pairs in each distance range bin
    count0=nd.zeros(binnum-1,ctx[0],dtype="float32")        
#Calculate index coordinates and directions by chuncks
    a=regions[i[0]*broadcdp:min((i[0]+1)*broadcdp,regions.shape[0]),:]
    b=regions[i[1]*broadcdp:min((i[1]+1)*broadcdp,regions.shape[0]),:]
    a1=nd.array(a,ctx[0])
    b1=nd.array(b,ctx[0])
    a1_b1=(nd.expand_dims(a1,axis=1)-b1).reshape((-1,2))
    x1_x2=a1_b1[:,0]
    y1_y2=a1_b1[:,1]
#Find the rows where all equal zeros and assign label -1
    boolmask=(x1_x2==0)*(y1_y2==0)
    labels=nd.zeros(boolmask.shape[0],ctx[0],dtype="float32")-1
    ldis=nd.broadcast_hypot(x1_x2,y1_y2).reshape((-1,))
#Change the zeros into -1
    ldis=nd.where(condition=boolmask,x=labels,y=ldis)
    for p in range (0,binnum-1):
        booleanmask=nd.equal((ldis>=bins[p]),(ldis<bins[p+1]))
        count0[p]+=nd.sum(booleanmask)
    return(count0.asnumpy())

#Full permutation distance computation between different regions: high and low
def distance11(regions_high,regions_low,i,binnum,bins,ctx):
#Initiate empty array for storing the number of counted pairs in each distance range bin
    count0=nd.zeros(binnum-1,ctx[0],dtype="float32")
#Calculate index coordinates and directions by chuncks
    a=regions_high[i[0]*broadcdp:min((i[0]+1)*broadcdp,regions_high.shape[0]),:]
    b=regions_low[i[1]*broadcdp:min((i[1]+1)*broadcdp,regions_low.shape[0]),:]
    a1=nd.array(a,ctx[0])
    b1=nd.array(b,ctx[0])
    a1_b1=(nd.expand_dims(a1,axis=1)-b1).reshape((-1,2))
    x1_x2=a1_b1[:,0]
    y1_y2=a1_b1[:,1]
#Find the rows where all equal zeros and assign label -1
    boolmask=(x1_x2==0)*(y1_y2==0)
    labels=nd.zeros(boolmask.shape[0],ctx[0],dtype="float32")-1
    ldis=nd.broadcast_hypot(x1_x2,y1_y2).reshape((-1,))
#Change the zeros into -1
    ldis=nd.where(condition=boolmask,x=labels,y=ldis)
    for p in range (0,binnum-1):
        booleanmask=nd.equal((ldis>=bins[p]),(ldis<bins[p+1]))
        count0[p]+=nd.sum(booleanmask)
    return(count0.asnumpy())

#Full combination distance computation
def distance2(regions,i,binnum,bins,ctx):
#Initiate empty array for storing the number of counted pairs in each distance range bin
    count0=nd.zeros(binnum-1,ctx[0],dtype="float32")
    seed=nd.zeros((1,2),ctx[0])
#Calculate index coordinates and directions by chuncks
    a=regions[i[0]*broadcdp:min((i[0]+1)*broadcdp,regions.shape[0]),:]
    b=regions[i[1]*broadcdp:min((i[1]+1)*broadcdp,regions.shape[0]),:]
    a1=nd.array(a,ctx[0])
    b1=nd.array(b,ctx[0])    
    for i in range (a1.shape[0]):
        if i<a1.shape[0]-1:
            a1_b1=(nd.expand_dims(a1[i].reshape((1,2)),axis=1)-b1[i+1:,:]).reshape((a1[i+1:,:].shape[0],2))
            seed=nd.concat(seed,a1_b1,dim=0)
    if seed.shape[0]>1:
        x1_x2=seed[:,0]
        y1_y2=seed[:,1]
#Find the rows where all equal zeros and assign label -1
        boolmask=(x1_x2==0)*(y1_y2==0)
        labels=nd.zeros(boolmask.shape[0],ctx[0],dtype="float32")-1
        ldis=nd.broadcast_hypot(x1_x2,y1_y2).reshape((-1,))
#Change the zeros into -1
        ldis=nd.where(condition=boolmask,x=labels,y=ldis)
        for p in range (0,binnum-1):
            booleanmask=nd.equal((ldis>=bins[p]),(ldis<bins[p+1]))
            count0[p]+=nd.sum(booleanmask)
    return(count0.asnumpy())

def topo(taoh_W,mean_d,cardh_his,taoh_W4,mean_d4,binnum):
#Compute TOPO
    OMNIW=np.zeros(binnum,dtype="float32")
    OMNIW4=np.zeros((4,binnum),dtype="float32")

    #Convert Nan to zero to avoid issues
    taoh_W1=np.nan_to_num(taoh_W)
    mean_d1=np.nan_to_num(mean_d)
    taoh_W41=np.nan_to_num(taoh_W4)
    mean_d41=np.nan_to_num(mean_d4)

    for j in range (binnum-1):
        if taoh_W1[j+1]!=0:
            OMNIW[0]+=(taoh_W1[j]+taoh_W1[j+1])*(mean_d1[j+1]-mean_d1[j])*0.5
            
    for k in range (4):
        for l in range (binnum-1):
            if taoh_W41[k,l+1]!=0:            
                OMNIW4[k,0]+=(taoh_W41[k,l]+taoh_W41[k,l+1])*(mean_d41[k,l+1]-mean_d41[k,l])*0.5

    results=np.vstack((taoh_W1,mean_d1,OMNIW,cardh_his))
    results4=np.vstack((taoh_W41,mean_d41,OMNIW4))
    return (results,results4)

def compu(expe,bins,dibins,dibins4,binnum,gt,rows,cols,flowdir,kernel_dict,ctx):
#Initiate empty array for storing histogram for directions, distances, and number of counted pairs in each distance range bin
    coAA=np.zeros((1,binnum-1),dtype="float32")
    codiAA=np.zeros((4,binnum-1),dtype="float32")
    countAA=np.zeros(binnum-1)
    countAZ=np.zeros(binnum-1)
    count4AA=np.zeros((4,binnum-1),dtype="float32")
    co4AA=np.zeros((4,binnum-1),dtype="float32")
    
#Create coordinate arrays for each zone and compute distances and directions
    #Area of High
    k=1
    regionA=np.asarray(np.where(expe>0),dtype="int32").T
    if regionA.shape[0]!=0:
        countA=itercontrol(regionA,k,bins,dibins,dibins4,binnum,rows,cols,flowdir,kernel_dict,ctx)
        k=0
        regionZ=np.asarray(np.where(expe==0),dtype="int32").T
        if regionZ.shape[0]!=0:
            countAZ=itercontrol((regionA,regionZ),k,bins,dibins,dibins4,binnum,rows,cols,flowdir,kernel_dict,ctx)
    #Each connected region in High
        k=2#Switch
        for i in range (1,np.amax(expe)+1):
            regionAA=np.asarray(np.where(expe==i),dtype="int32").T
            outAA=itercontrol(regionAA,k,bins,dibins,dibins4,binnum,rows,cols,flowdir,kernel_dict,ctx)
            coAA+=outAA[0];codiAA+=outAA[1];countAA+=outAA[2];co4AA+=outAA[3];count4AA+=outAA[4]

#Compute connectivity metrics
    #Add the first element
    taoh_W=np.append(1,(countAA/(countA+countAZ)))#
    #Average connected distances in each range bin
    mean_d=np.append(0,(coAA*gt[1]/countAA))
    #Histogram of connected directions (4 total fr om East) for each range bin
    cardh_his=np.append(np.zeros((4,1),dtype="float32")+regionA.shape[0],codiAA,axis=1)
    taoh_W4=np.append(np.zeros((4,1),dtype="float32")+1,count4AA/(countA+countAZ),axis=1)
    mean_d4=np.append(np.zeros((4,1),dtype="float32"),co4AA*gt[1]/count4AA,axis=1)   
    return (taoh_W,mean_d,cardh_his,taoh_W4,mean_d4)

def prires(results,results4,bins,gt):
#Print out results as Pandas dataframe and write to text files
    rowlabel=np.array(["taoh_W","mean_distance","OMNIW","CARD_Histogram_WE",
                       "NE_SW","NS","NW_SE"]).reshape(7,1)
    colabel=np.empty(binnum,dtype="U30")
    binslabel=np.around(bins*gt[1], decimals=3)
    for i in range(binnum-1):
        colabel[i+1]="Lag "+str(binslabel[i])+"-"+str(binslabel[i+1])
    colabel[0]="Lag 0"
    results_df=pd.DataFrame(results,columns=colabel)
    results_df.insert(0, "Variables", rowlabel)
    results_df=results_df.round(6)
    rowlabel4=np.array(["taoh_W_WE","taoh_W_NE_SW","taoh_W_NS","taoh_W_NW_SE",
                        "mean_distance_WE","mean_distance_NE_SW","mean_distance_NS","mean_distance_NW_SE",
                        "OMNIW_WE","OMNIW_NE_SW","OMNIW_NS","OMNIW_NW_SE",
                        ]).reshape(12,1)
    results_df4=pd.DataFrame(results4,columns=colabel)
    results_df4.insert(0, "Variables", rowlabel4)
    results_df4=results_df4.round(6)
    return (results_df,results_df4)


if __name__ == '__main__':
#Set variables
    broadcdp=1700
    threshold=0.1
    NoData=-999
    binnum=20
    ctx=[gpu(0)]    
    
#Initiate bin for 4 cardinal directions in reach distance range bin
    dibins=np.array([0,22.5,67.5,112.5,157.5,181])
#Initiate bin for using 4 cardinal directions to extract connectivity functions
    dibins4=np.array([0,22.5,67.5,112.5,157.5,181])
    
#Input files and parameters for DEM
    filenameFlow='flowpatterninput'
    path = os.path.join('inputdirectory', filenameFlow)
    ds = gdal.Open(path)
    gt = ds.GetGeoTransform()
    flowpattern = np.array(ds.GetRasterBand(1).ReadAsArray())#
    rows=flowpattern.shape[0]
    cols=flowpattern.shape[1]
    maxd=((rows-1)**2+(cols-1)**2)**0.5
    bins=np.linspace(1,maxd,num=binnum,endpoint=True)

#Input files and parameters for flow patterns
    filenameDEM='deminput'
    path = os.path.join('inputdirectory', filenameDEM)
    ds = gdal.Open(path)
    gt = ds.GetGeoTransform()
    dem_ori = np.array(ds.GetRasterBand(1).ReadAsArray(),dtype="float32")#
    dem_ori[dem_ori<0]=NoData
    start = timer()

    flowdir_amend,edgecells,pourpoints,kernel_dict=demprocess(dem_ori,NoData,rows,cols,ctx)
    flowpattern=prep(flowpattern,threshold,NoData)    
    results,results4=topo(*compu(flowpattern,bins,dibins,dibins4,binnum,gt,rows,cols,flowdir_amend,kernel_dict,ctx),binnum)
    results_df,results4_df=prires(results,results4,bins,gt)

    end = timer()
    
#Save results to txt files
    results_df.to_csv(filenameFlow+filenameDEM+"results_taoh.csv", index=True, header=True)
    results4_df.to_csv(filenameFlow+filenameDEM+"results_CARD.csv", index=True, header=True)
    np.savetxt(filenameFlow+filenameDEM+"computingtime.csv",[end-start],delimiter=",")
