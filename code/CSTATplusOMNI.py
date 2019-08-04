#CSTAT+ A GPU-accelerated spatial pattern analysis algorithm for high-resolution 2D/3D hydrologic connectivity using array vectorization and convolutional neural network 
#Author: Feng Yu, Jonathan M. Harbor
#Department of Earth, Atmospheric and Planetary Sciences, Purdue University, 550 Stadium Mall Dr, West Lafayette, IN 47907 USA.
#Email: yu172@purdue.edu; Alternative: fyu18@outlook.com
#This is the omnidirectional version: CSTAT+/OMNI

import os
from osgeo import gdal 
import numpy as np    
import copy as cp
from numpy import genfromtxt as gft
from scipy.ndimage.measurements import label
from itertools import combinations_with_replacement,product
from mxnet import nd,gpu          
from timeit import default_timer as timer
import pandas as pd
 
#Binarize pattern
def prep(expe0,threshold,NoData):
    #Provide threshold for High/Low, usually the depth of shallow sheetflow
    expe1=cp.deepcopy(expe0)
    expe2=cp.deepcopy(expe0)
    expe1[(expe1>=threshold)]=1
    expe1[(expe1<threshold)]=0
    expe2[(expe2==NoData)]=-1
    expe2[(expe2>0)]=0
    connection_structure = np.array([[1,1,1],[1,1,1],[1,1,1]])
    expela, num_features =label (expe1,structure=connection_structure)
    expe3=expe2+expela
    return (expe3)

def itercontrol(regions,k,bins,dibins,dibins4,binnum):
#Initiate empty array for storing histogram for directions, distances, and number of counted pairs in each distance range bin
    co0=nd.zeros(binnum-1,gpu(0),dtype="float32")
    codi0=nd.zeros((4,binnum-1),gpu(0),dtype="float32")
    count0=nd.zeros(binnum-1,gpu(0),dtype="float32")
    count4=nd.zeros((4,binnum-1),gpu(0),dtype="float32")
    co4=nd.zeros((4,binnum-1),gpu(0),dtype="float32")
    bins=nd.array(bins,gpu(0))
    dibins=nd.array(dibins,gpu(0))
    dibins4=nd.array(dibins4,gpu(0))   
    if k==2:
#Create segment index for the input array to meet the memory requirement
        imax=list(range(int(regions.shape[0]/broadcdp)+(regions.shape[0]%broadcdp!=0)))

#Combinations with repeated indicies
        iterator=list(combinations_with_replacement(imax,2))
        for i in iterator:
            if i[0]==i[1]:
                vout=distanceAA2(regions,i,binnum,dibins,dibins4)
                co0+=vout[0]
                codi0+=vout[1]
                count0+=vout[2]
                co4+=vout[3]
                count4+=vout[4]
            else:
                vout=distanceAA1(regions,i,binnum,dibins,dibins4)
                co0+=vout[0]
                codi0+=vout[1]
                count0+=vout[2]
                co4+=vout[3]
                count4+=vout[4]                
        return (co0.asnumpy(),codi0.asnumpy(),count0.asnumpy(),co4.asnumpy(),count4.asnumpy())
    elif k==1:
#Create segment index for the input array to meet the memory requirement
        imax=list(range(int(regions.shape[0]/broadcdp)+(regions.shape[0]%broadcdp!=0)))

#Combinations with repeated indicies
        iterator=list(combinations_with_replacement(imax,2))
        for i in iterator:
            if i[0]==i[1]:
                count0+=distance2(regions,i,binnum,bins)    
            else:
                count0+=distance1(regions,i,binnum,bins)
        return (count0.asnumpy())
    else:
#Unpack the tuple
        regions_high,regions_low=regions        
#Create segment index for the input array to meet the memory requirement
        imax_high=list(range(int(regions_high.shape[0]/broadcdp)+(regions_high.shape[0]%broadcdp!=0)))
        imax_low=list(range(int(regions_low.shape[0]/broadcdp)+(regions_low.shape[0]%broadcdp!=0)))
#Combinations with repeated indicies
        iterator=list(product(imax_high,imax_low))
        for i in iterator:
            count0+=distance11(regions_high,regions_low,i,binnum,bins)
        return (count0.asnumpy())

def distanceAA1(regions,i,binnum,dibins,dibins4):
#Initiate empty array for storing histogram for directions, distances, and number of counted pairs in each distance range bin
    co0=nd.zeros(binnum-1,gpu(0),dtype="float32")
    codi0=nd.zeros((5,binnum-1),gpu(0),dtype="float32")
    count0=nd.zeros(binnum-1,gpu(0),dtype="float32")
    count4=nd.zeros((5,binnum-1),gpu(0),dtype="float32")
    co4=nd.zeros((5,binnum-1),gpu(0),dtype="float32")
    
#Calculate index coordinates and directions by chuncks
    a=regions[i[0]*broadcdp:min((i[0]+1)*broadcdp,regions.shape[0]),:]
    b=regions[i[1]*broadcdp:min((i[1]+1)*broadcdp,regions.shape[0]),:]
    a1=nd.array(a,gpu(0))
    b1=nd.array(b,gpu(0))
    a1_b1=(nd.expand_dims(a1,axis=1)-b1).reshape((-1,2))
    x1_x2=a1_b1[:,0]
    y1_y2=a1_b1[:,1]
    labels=nd.zeros(x1_x2.shape[0],gpu(0),dtype="float32")
    sdi0=(nd.degrees(nd.arctan((y1_y2)/(x1_x2)))+90).reshape((-1,))
    ldis=nd.broadcast_hypot(x1_x2,y1_y2).reshape((-1,))
#Change 0 to 180 so it can apply sum of boolean mask without losing values        
    sdi0=nd.where(condition=(sdi0==0),x=labels+180,y=sdi0)
#Store sum of distances co0 and histogram of directions in each range bin
    for p in range (0,binnum-1):
        booleanmask=nd.equal((ldis>=bins[p]),(ldis<bins[p+1]))
        count0[p]+=nd.nansum(booleanmask)
        co0[p]+=nd.nansum(ldis*booleanmask)
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
            count4[k,l]+=nd.nansum(booleanmask)
            co4[k,l]+=nd.nansum(ldis0*booleanmask)
            
    codi0[0,:]+=codi0[4,:]
    codi0=codi0[0:4,:]
    count4[0,:]+=count4[4,:]
    count4=count4[0:4,:]
    co4[0,:]+=co4[4,:]
    co4=co4[0:4,:]
    return(co0,codi0,count0,co4,count4)

def distanceAA2(regions,i,binnum,dibins,dibins4):
#Initiate empty array for storing histogram for directions, distances, and number of counted pairs in each distance range bin
    co0=nd.zeros(binnum-1,gpu(0),dtype="float32")
    codi0=nd.zeros((5,binnum-1),gpu(0),dtype="float32")
    count0=nd.zeros(binnum-1,gpu(0),dtype="float32")
    count4=nd.zeros((5,binnum-1),gpu(0),dtype="float32")
    co4=nd.zeros((5,binnum-1),gpu(0),dtype="float32")
    seed=nd.zeros((1,2),gpu(0))
#Calculate index coordinates and directions by chuncks
    a=regions[i[0]*broadcdp:min((i[0]+1)*broadcdp,regions.shape[0]),:]
    b=regions[i[1]*broadcdp:min((i[1]+1)*broadcdp,regions.shape[0]),:]
    a1=nd.array(a,gpu(0))
    b1=nd.array(b,gpu(0))
#    print ("a1",a1,"b1",b1)
    for ii in range (a1.shape[0]-1):
        a1_b1=(nd.expand_dims(a1[ii].reshape((1,2)),axis=1)-b1[ii+1:,:]).reshape((a1[ii+1:,:].shape[0],2))
        seed=nd.concat(seed,a1_b1,dim=0)
    if seed.shape[0]>1:
        x1_x2=seed[1:,0]
        y1_y2=seed[1:,1]
        labels=nd.zeros(x1_x2.shape[0],gpu(0),dtype="float32")
        sdi0=(nd.degrees(nd.arctan((y1_y2)/(x1_x2)))+90).reshape((-1,))
        ldis=nd.broadcast_hypot(x1_x2,y1_y2).reshape((-1,))

#Change 0 to 180 so it can apply sum of boolean mask without losing values        
        sdi0=nd.where(condition=(sdi0==0),x=labels+180,y=sdi0)

#Store sum of distances co0 and histogram of directions in each range bin
        for p in range (0,binnum-1):
            booleanmask=nd.equal((ldis>=bins[p]),(ldis<bins[p+1]))
            count0[p]+=nd.nansum(booleanmask)
            co0[p]+=nd.nansum(ldis*booleanmask)

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
                count4[k,l]+=nd.nansum(booleanmask)
                co4[k,l]+=nd.nansum(ldis0*booleanmask)

    codi0[0,:]+=codi0[4,:]
    codi0=codi0[0:4,:]
    count4[0,:]+=count4[4,:]
    count4=count4[0:4,:]
    co4[0,:]+=co4[4,:]
    co4=co4[0:4,:]
    return(co0,codi0,count0,co4,count4)

#Full permutation distance computation
def distance1(regions,i,binnum,bins):
#Initiate empty array for storing the number of counted pairs in each distance range bin
    count0=nd.zeros(binnum-1,gpu(0),dtype="float32")        

#Calculate index coordinates and directions by chuncks
    a=regions[i[0]*broadcdp:min((i[0]+1)*broadcdp,regions.shape[0]),:]
    b=regions[i[1]*broadcdp:min((i[1]+1)*broadcdp,regions.shape[0]),:]
    a1=nd.array(a,gpu(0))
    b1=nd.array(b,gpu(0))
    a1_b1=(nd.expand_dims(a1,axis=1)-b1).reshape((-1,2))
    x1_x2=a1_b1[:,0]
    y1_y2=a1_b1[:,1]
    ldis=nd.broadcast_hypot(x1_x2,y1_y2).reshape((-1,))
    for p in range (0,binnum-1):
        booleanmask=nd.equal((ldis>=bins[p]),(ldis<bins[p+1]))
        count0[p]+=nd.nansum(booleanmask)
    return(count0)

#Full permutation distance computation between different regions: high and low
def distance11(regions_high,regions_low,i,binnum,bins):
#Initiate empty array for storing the number of counted pairs in each distance range bin
    count0=nd.zeros(binnum-1,gpu(0),dtype="float32")
#Calculate index coordinates and directions by chuncks
    a=regions_high[i[0]*broadcdp:min((i[0]+1)*broadcdp,regions_high.shape[0]),:]
    b=regions_low[i[1]*broadcdp:min((i[1]+1)*broadcdp,regions_low.shape[0]),:]
    a1=nd.array(a,gpu(0))
    b1=nd.array(b,gpu(0))
    a1_b1=(nd.expand_dims(a1,axis=1)-b1).reshape((-1,2))
    x1_x2=a1_b1[:,0]
    y1_y2=a1_b1[:,1]
    ldis=nd.broadcast_hypot(x1_x2,y1_y2).reshape((-1,))
    for p in range (0,binnum-1):
        booleanmask=nd.equal((ldis>=bins[p]),(ldis<bins[p+1]))
        count0[p]+=nd.nansum(booleanmask)
    return(count0)

#Full combination distance computation
def distance2(regions,i,binnum,bins):
#Initiate empty array for storing the number of counted pairs in each distance range bin
    count0=nd.zeros(binnum-1,gpu(0),dtype="float32")
    seed=nd.zeros((1,2),gpu(0))
#Calculate index coordinates and directions by chuncks
    a=regions[i[0]*broadcdp:min((i[0]+1)*broadcdp,regions.shape[0]),:]
    b=regions[i[1]*broadcdp:min((i[1]+1)*broadcdp,regions.shape[0]),:]
    a1=nd.array(a,gpu(0))
    b1=nd.array(b,gpu(0))    
    for ii in range (a1.shape[0]-1):
        a1_b1=(nd.expand_dims(a1[ii].reshape((1,2)),axis=1)-b1[ii+1:,:]).reshape((a1[ii+1:,:].shape[0],2))
        seed=nd.concat(seed,a1_b1,dim=0)
    if seed.shape[0]>1:
        x1_x2=seed[1:,0]
        y1_y2=seed[1:,1]
        ldis=nd.broadcast_hypot(x1_x2,y1_y2).reshape((-1,))
        for p in range (0,binnum-1):
            booleanmask=nd.equal((ldis>=bins[p]),(ldis<bins[p+1]))
            count0[p]+=nd.nansum(booleanmask)
    return(count0)

def omni(taoh_W,mean_d,cardh_his,taoh_W4,mean_d4,binnum):
#Compute OMNI
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

def compu(flowpattern,bins,dibins,dibins4,binnum,gt):
#Initiate empty array for storing histogram for directions, distances, and number of counted pairs in each distance range bin
    coAA=np.zeros((1,binnum-1),dtype="float32")
    codiAA=np.zeros((4,binnum-1),dtype="float32")
    countAA=np.zeros(binnum-1)
    countAZ=np.zeros(binnum-1)
    count4AA=np.zeros((4,binnum-1),dtype="float32")
    co4AA=np.zeros((4,binnum-1),dtype="float32")    
#Create coordinate arrays for each zone and compute distances and directions
    #All the domain area excluding NoData
    #Area of High
    k=1
    regionA=np.asarray(np.where(flowpattern>0),dtype="int32").T
    if regionA.shape[0]!=0:
        countA=itercontrol(regionA,k,bins,dibins,dibins4,binnum)
        k=0
        regionZ=np.asarray(np.where(flowpattern==0),dtype="int32").T
        if regionZ.shape[0]!=0:
            countAZ=itercontrol((regionA,regionZ),k,bins,dibins,dibins4,binnum)
    #Each connected region in High
        k=2#Switch
        for i in range (1,np.int32(np.amax(flowpattern)+1)):
            regionAA=np.asarray(np.where(flowpattern==i),dtype="int32").T
            outAA=itercontrol(regionAA,k,bins,dibins,dibins4,binnum)
            coAA+=outAA[0];codiAA+=outAA[1];countAA+=outAA[2];co4AA+=outAA[3];count4AA+=outAA[4]

#Compute connectivity metrics
    if np.sum(countAZ)==0:
        taoh_W=np.append(1,(countAA/(countA+countAZ)))#;taoh_M=np.append((regionA.shape[0]/regionZ.shape[0]),(countAA/countZ))
    else:
        taoh_W=np.append(1,(countAA*2/(countA+countAZ)))  
    #Average connected distances in each range bin
    mean_d=np.append(0,(coAA*gt[1]/countAA))
    #Histogram of connected directions (4 total fr om East) for each range bin
    cardh_his=np.append(np.zeros((4,1),dtype="float32")+regionA.shape[0],codiAA,axis=1)
    #Tao(h) and Average connected distances in each cardinal direction (4 total: W-E, NE-SW, N-S, NW-SE)
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
    broadcdp=1500
    threshold=0.5
    NoData=0
    binnum=20

#Initiate bin for 4 cardinal directions in reach distance range bin
    dibins=np.array([0,22.5,67.5,112.5,157.5,181])
#Initiate bin for using 4 cardinal directions to extract connectivity functions
    dibins4=np.array([0,22.5,67.5,112.5,157.5,181])

#Input files and parameters
    filename='inputfilename'
    path = os.path.join('inputdirectory', filename)
    ds = gdal.Open(path)
    gt = ds.GetGeoTransform()
    expe = np.array(ds.GetRasterBand(1).ReadAsArray(),dtype="float32")#
    rows=expe.shape[0]
    cols=expe.shape[1]
    maxd=((rows-1)**2+(cols-1)**2)**0.5
    bins=np.linspace(1,maxd,num=binnum,endpoint=True)
    start = timer()

    flowpattern=prep(expe,threshold,NoData)
    results,results4=omni(*compu(flowpattern,bins,dibins,dibins4,binnum,gt),binnum)
    results_df,results4_df=prires(results,results4,bins,gt)

    end = timer()

#Save results to txt files
    results_df.to_csv(filename+"results_taoh.csv", index=True, header=True)
    results4_df.to_csv(filename+"results_CARD.csv", index=True, header=True)
    np.savetxt(filename+"computingtime.csv",[end-start],delimiter=",")


