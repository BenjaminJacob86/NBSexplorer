"""
test
"""
#docker exec -it nbs-explorer-job-aws bash
## modules ##########################
## Plots
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

#### for SCHISM
import sys
import os
from schism import*

#cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable # allow axis adjustment of colorbar	  

from scipy.spatial import cKDTree
import yaml

# custom colormaps
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
import s3fs   # To access files on user dir directly using AWS
import glob  # crasht bei glob
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
#########################################


# selection widdgets
cwd=os.getcwd()

userdir=os.environ.get("EDITO_INFRA_OUTPUT")  
########## temp test ############

mk_plots=os.environ.get("MAKE_PLOTS")
saveimg=mk_plots=='true' #True

localtest=False
if localtest:
    saveimg=True

# s3fs access:
if not localtest:
    AWS_S3_ENDPOINT=os.environ.get("AWS_S3_ENDPOINT")  
    AWS_ACCESS_KEY_ID=os.environ.get("AWS_ACCESS_KEY_ID")  
    AWS_SECRET_ACCESS_KEY=os.environ.get("AWS_SECRET_ACCESS_KEY")  
    AWS_SESSION_TOKEN=os.environ.get("AWS_SESSION_TOKEN")  

    # access user my Files directory
    userMyFilesdir=os.environ.get("User_Name").replace('user','oidc')  # get user name from environment variabl user-jacobb

    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': 'https://'+'minio.lab.dive.edito.eu'},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        token=os.environ["AWS_SESSION_TOKEN"])

##### Copy ncessary files for container ############################

YAML_FILE=os.environ.get("YAML_FILE")  


if not localtest:
    outpt =  os.path.join(userdir,'output.txt') # join directory userdir not mpy
    processed =  os.path.join(userdir,'input_processed.txt')
    log =  os.path.join(userdir,'log.txt')
    lslog =  os.path.join(userdir,'lslog.txt')
    np.savetxt(outpt,[1,2,3]) # write
else:
    userdir='./'
    lslog =  cwd+'/lslog.txt'

# Open Yaml File:
if not localtest:
    f=fs.open(userMyFilesdir+'/nbsconfig.yaml', "r")
    config = yaml.safe_load(f)
else:    
    f=open('./nbsconfig.yaml')
    config = yaml.safe_load(f)
    
# Load yaml file    
SLR=config['options']['scenario']['SLR']
# Define Area
LONmin=config['options']['area']['min_lon']
LONmax=config['options']['area']['max_lon']
LATmin=config['options']['area']['min_lat']
LATmax=config['options']['area']['max_lat']
bbox=np.asarray( [[LONmin,LATmin],[LONmax,LATmax]])

# define plant characteristics
Dstem=config['options']['Plant Physiology']['stem_diameter']
Cdrag=config['options']['Plant Physiology']['drag_coefficient']
Hcanopy=config['options']['Plant Physiology']['canopy_height']

# def # assign nbs
# invert postiv downard definitions
Dupper=-config['options']['positioning']['upper_limit']
Dlower=-config['options']['positioning']['lower_limit']
##################################

################## load schism Setup
setupdir='./grid/'
os.chdir(setupdir)
s0=schism_setup() #second instances for overlayed plotting with mask
s=schism_setup()
os.chdir(cwd)
s.lon,s.lat=np.asarray(s.lon),np.asarray(s.lat)
s.x,s.y=np.asarray(s.x),np.asarray(s.y)
s.depths=np.asarray(s.depths)

#get Areas
A=[]
for i in range(s.nvplt.shape[0]):
    nodes=s.nvplt[i,:]+1
    A.append(s.proj_area(nodes))
A=np.asarray(A)

faces=s.nvplt
x=np.asarray(s.lon)
y=np.asarray(s.lat) 

# extent_total_ares
xmin,ymin=np.min(s.lon),np.min(s.lat)
xmax,ymax=np.max(s.lon),np.max(s.lat)
##############################################


###### functions #####
def plot_bg_map(ax,LONmin,LONmax,LATmin,LATmax):
    proj=ccrs.Mercator()  # define Prijection

    land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',edgecolor='face',facecolor=cfeature.COLORS['land'])
    ocean_10m = cfeature.NaturalEarthFeature('physical', 'ocean', '10m', edgecolor='face',facecolor=cfeature.COLORS['water']) #facecolor=[0,0,1]

    zoom_extend=(LONmin,LONmax, LATmin,LATmax)
    ax.set_extent(zoom_extend)
    ax.add_feature(land_10m,zorder=-2)
    ax.add_feature(ocean_10m,zorder=-2)

    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)		  

    longrange=LONmax-LONmin
    latrange=LATmax-LATmin
    ratio=latrange/longrange

    if ratio > 1:
        nxticks=int(np.floor(8/ratio))
        nxticks=5
    else:
        nxticks=8
        #nyticks=int(np.floor(8*ratio))
        nyticks=int(np.ceil(8*ratio))
    nxticks=5
    nyticks=5
      
    xticks=np.unique(np.round(np.linspace(LONmin,LONmax,nxticks),2))
    yticks=np.unique(np.round(np.linspace(LATmin,LATmax,nyticks),2))

    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(number_format='.1f',degree_symbol='',dateline_direction_label=True)
    lat_formatter = LatitudeFormatter(number_format='.1f',degree_symbol='')
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

def plot_bbox(ax,bbox):
    ax.plot(bbox[[0,1,1,0,0],0],bbox[[0,0,1,1,0],1],'r') 

def create_segmented_colormap(base_cmap, num_segments):
    # Get the existing colormap
    cmap = plt.get_cmap(base_cmap)

    # Determine the number of colors to sample from the existing colormap
    num_colors = 256  # Number of colors in jet colormap
    indices = np.linspace(0, num_colors - 1, num_segments, dtype=int)

    # Sample colors from the existing colormap
    colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
    sampled_colors = [colors[i] for i in indices]

    # Create a segmented colormap using LinearSegmentedColormap
    return mcolors.LinearSegmentedColormap.from_list(f'{base_cmap}_segmented', sampled_colors, N=num_segments)


def imscatter(x, y, image='./SGplant.png', ax=None, zoom=1):
""" Plot an image inside axis"""
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


##################################

### Define seagrass positoning based on beach profile dpeths ###
DInds=(s.depths >= Dupper) & (s.depths <= Dlower)
geoInds=(s.lon >= LONmin) & (s.lon <= LONmax) & (s.lat >=LATmin) & (s.lat <= LATmax)
use1=np.zeros(s.nnodes,bool)
use1[geoInds] = 1

plotnodes=use1
plotelems=use1[s.nvplt].min(axis=1)

#where NBS
use2=np.zeros(s.nnodes,bool)
use2[geoInds & DInds] = 1
NBSnodes=use2
NBSelems=use2[s.nvplt].min(axis=1)
########################

# second grid instance for transparaency plotting
mask=np.zeros(s0.nvplt.shape,bool)
mask[NBSelems==False,:]=True
s0.nvplt=np.ma.masked_array(s0.nvplt,mask=mask)
s0.nvplt=s.nvplt[NBSelems,:]  # remove from second plotting grid to hav bathy in background

# kein cartopy plot eventuell?
dpi=200 # image resolution 
 
 
###### Beach Profile plot ####
Dmax=-Dupper
Dmin=-Dlower
M=np.loadtxt('beachCoords.ascii')
bx,by=M[:,0],M[:,1]

fig = plt.figure(figsize=(4.5, 3), dpi=dpi) # main plot
axNBS=fig.add_subplot(111) 
axNBS.set_ylim((-8,8))
axNBS.set_xlim((0,34))
axNBS.plot(bx,by,'y-',linewidth=3)
axNBS.fill(bx[[0,-1,-1,0]],[0,0,-8,-8],color='aqua')
axNBS.fill(np.hstack((bx,[0,])),np.hstack((by,[by.min(),])),color='gold')
axNBS.set_aspect(7/8)
axNBS.set_xlabel('Beach Width')
axNBS.set_ylabel('Height [m]')
axNBS.set_title('Illustrative Beach Crossection')
axNBS.set_ylim((-8,8))
axNBS.set_xlim((0,34))
fig.tight_layout()

for xq in np.arange(0,34,0.8):
    nn=np.argmin(np.abs(bx-xq))
    if (by[nn]>Dmin) & (by[nn]<Dmax):
        imscatter(bx[nn], by[nn]+1.75, './SGplant.png', ax=axNBS, zoom=0.65)[0]
figname='{:s}__SG{:.1f}-{:.1f}.png'.format(Dupper,Dlower,'SeagrassBeachCoverage')
figname=os.path.join(userdir,figname)
plt.savefig(figname,dpi=dpi)
plt.close()
############### / Beach plot ##############################

 
 
# plot domain and seagrass coverage Seagrass 
if saveimg:  # laeuft uber 20 min
    fig,ax = plt.subplots(figsize=(5, 3), dpi=dpi,ncols=1, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent((LONmin-0.01,LONmax+0.01,LATmin-0.01,LATmax+0.01))
    ph,ch,ax=s.plotAtnodesGeo(s.depths,proj='Carree',ax=ax,mask=plotelems !=1,cmap=plt.cm.Greys)
    ch.remove()
    ph,ch,ax=s0.plotAtnodesGeo(1.0*NBSnodes,proj='Carree',ax=ax,cmap=plt.cm.Greens)
    ch.remove()
    plt.title('Seagrass')
    ax.set_extent((LONmin-0.01,LONmax+0.01,LATmin-0.01,LATmax+0.01))
    plot_bbox(ax,bbox)
    plt.tight_layout()
    figname='{:.01f}E_{:.01f}E_{:.0f}N_{:.1f}_SG{:.1f}-{:.1f}_{:s}.png'.format(LONmin,LONmax,LATmin,LATmax,Dupper,Dlower,'segrassPositioning')
    figname=os.path.join(userdir,figname)
    plt.savefig(figname,dpi=dpi)

# results
# Plot Risk Map
# Plot Wave Height
# Plot Sediment Map



#try:
#    os.chdir('./results/SCENARIO/') # In container
#    files=glob.glob('*.nc')
#    with open(lslog ,'w') as f:
#        f.write(files)
#except:
#    with open(lslog ,'w') as f:
#        f.write('error writing files')


os.chdir(cwd)
files=glob.glob('results/SCENARIO/*.nc')
with open(lslog ,'w') as f:
    f.write('files in results glob')    
    f.write(str(files))
with open(lslog ,'a') as f:
    f.write('ls gives:' + str(os.listdir('./')))
    #os.listdir('results')
    #os.listdir('results/SCENARIO/')
#
files=[file[file.rindex('/')+1:] for file in files] # files drop path
files=[file for file in files if ('X' not in file) and ('Y' not in file)]
shortconfig=[file.split('_')[0] for file in files]

varlist=shortconfig
loaddict={}
for key,val in zip(shortconfig,files):
    loaddict[key]=val
plotvars=['totalSuspendedLoad','sigWaveHeight']
units={'totalSuspendedLoad':'[g/L]','sigWaveHeight':'[m]'}

# docker run --volume=$(pwd):/app nbs-explorer-job-aws

plotdata=np.zeros(s.nnodes) # weight witn in
tree = cKDTree(list(zip(s.x[NBSnodes],s.y[NBSnodes])))

for  plotvar in plotvars:
    print('loading ' + loaddict[plotvar])

    ds0=xr.open_dataset('results/CNTRL/'+loaddict[plotvar])
    ds1=xr.open_dataset('results/SCENARIO/'+loaddict[plotvar])

    myvar=list(ds1.variables)[0]
    temp=myvar.split('_')
    delta=(ds1[myvar]-ds0[myvar]).values
    plotdata=np.zeros(s.nnodes) # weight witn increasing distanes from neasrest NBS

    # interp results beteen max scenario und reference
    qinds=np.where(plotnodes)[0]
    dists,nn=tree.query(list(zip(s.x[qinds],s.y[qinds])))
    w=np.exp(-dists/100)
    plotdata[plotnodes]=delta[plotnodes]*w # weight

    plt.close('all')
    ph,ch,ax=s.plotAtnodesGeo(plotdata,proj='Carree',cmap=plt.cm.RdBu_r) #,ax=ax
    ax.set_title('Seagrass effect on '+ temp[0])
    unit=units[plotvar]
    label=temp[0] + ' change ' + unit
    vmax=np.nanquantile(np.abs(plotdata[NBSnodes]),0.95)
    ph.set_clim(-vmax,vmax)
    ch.set_label(label)
    ax.set_extent((LONmin-0.01,LONmax+0.01,LATmin-0.01,LATmax+0.01))

    plt.tight_layout()
    figname='{:.01f}E_{:.01f}E_{:.0f}N_{:.1f}_SG{:.1f}-{:.1f}_{:s}.png'.format(LONmin,LONmax,LATmin,LATmax,Dupper,Dlower,temp[0])
    figname=os.path.join(userdir,figname)
    plt.savefig(figname,dpi=dpi)


##### worked until here ################


##### Risk maps
plotvar='CriticalBottomStressRatio.nc'
ds0=xr.open_dataset('results/CNTRL/'+loaddict[plotvar])
ds1=xr.open_dataset('results/SCENARIO/'+loaddict[plotvar])

#interpolated for measure
R1=ds0['CriticalBottomStressRatio'].values
R2=R1.copy()
R2[plotnodes]=ds1['CriticalBottomStressRatio'].values[plotnodes]*w+R1[plotnodes]*(1-w)

green=np.array([0/256,128/256,0/256,1])
yellow=np.array([255/256,255/256,128/256,1])
orange=np.array([255/256,165/256,0/256,1])
red=np.array([256/256,0/256,0/256,1])
clrs=np.vstack((green,yellow,orange,red))
mymap = ListedColormap(clrs)  # reguster colormap for matplotlib

### interpolation
zoom_extend=(LONmin,LONmax, LATmin,LATmax)
plt.close()
ph,ch,ax=s.plotAtnodesGeo(R1,cmap=mymap,proj='Carree',region_limit=zoom_extend,extend='neither') #,extend='None'
ax.set_xlim((LONmin-0.01,LONmax+0.01))
ax.set_ylim((LATmin-0.01,LATmax+0.01))
ph.set_clim((0,1)) #
ch.set_ticks(np.linspace(0,.75,4))
ch.set_ticklabels(['No','low','increased','high'])
plt.sca(ax)
plt.title('Erosion Risk - Reference')
plt.tight_layout()
figname='{:.01f}E_{:.01f}E_{:.0f}N_{:.1f}_SG{:.1f}-{:.1f}_{:s}_default_ErosionRisk.png'.format(LONmin,LONmax,LATmin,LATmax,Dupper,Dlower,temp[0])
figname=os.path.join(userdir,figname)
plt.savefig(figname,dpi=dpi)

## NBS
plt.close()
ph,ch,ax=s.plotAtnodesGeo(R2,cmap=mymap,proj='Carree',region_limit=zoom_extend,extend='neither') 
ax.set_xlim((LONmin-0.01,LONmax+0.01))
ax.set_ylim((LATmin-0.01,LATmax+0.01))
ph.set_clim((0,1)) #
ch.set_ticks(np.linspace(0,.75,4))
ch.set_ticklabels(['No','low','increased','high'])
plt.sca(ax)
plt.title('Erosion Risk - With NbS')
plt.tight_layout()
figname='{:.01f}E_{:.01f}E_{:.0f}N_{:.1f}_SG{:.1f}-{:.1f}_{:s}_nbs_ErosionRisk.png'.format(LONmin,LONmax,LATmin,LATmax,Dupper,Dlower,temp[0])
figname=os.path.join(userdir,figname)
plt.savefig(figname,dpi=300)

### bin into intervalls for risk level reduction
bins=[0,0.25,.5,.75]
R1bins=R1.copy()
R2bins=R2.copy()
for lbin in bins:
    gt = R1 >  lbin
    R1bins[gt] =lbin
    gt = R2 >  lbin
    R2bins[gt] =lbin
# nomralize toi levels 1,2,3    
R1bins/=0.25  
R2bins/=0.25  



## Example usage:
base_cmap = 'seismic'  # 'jet' Existing colormap
num_segments = 7  # Number of segments for the new colormap
segmented_cmap = create_segmented_colormap(base_cmap, num_segments)
#

new_cmap=segmented_cmap 
plt.close()
ph,ch,ax=s.plotAtnodesGeo(R2bins-R1bins,proj='Carree',cmap=new_cmap,region_limit=zoom_extend) #,extend='None'
ax.set_xlim((LONmin-0.01,LONmax+0.01))
ax.set_ylim((LATmin-0.01,LATmax+0.01))
ph.set_clim((-3,3))
plt.sca(ax)
plt.title('Erosion Risk level change due to NBS')
plt.tight_layout()
figname='{:.01f}E_{:.01f}E_{:.0f}N_{:.1f}_SG{:.1f}-{:.1f}_{:s}_nbs_ErosionRiskReduction.png'.format(LONmin,LONmax,LATmin,LATmax,Dupper,Dlower,temp[0])
figname=os.path.join(userdir,figname)
plt.savefig(figname,dpi=dpi)

print('done')