"""
NBS configuration GUI code for the EDITO Digital twin project.
Different GUI subitems for configuration and exploration are loaded
as subpages to the main frame. Variables needed for exhange between the subpages
to are  configures as item of controller (e.g. Controller.SLR).
Items such as buttons are currently lazily automatically position using pack wherfigname='{:.01f}E_{:.01f}E_{:.0f}N_{:.1f}_SG{:.1f}-{:.1f}_{:s}.png'.e it is considered sutiable.


"""


# try environment and:
#https://www.youtube.com/watch?v=JDU-ycsxvqM
# install custom tkinter


# NoN - GUI

# von pack auf grid?

## modules ##########################
import tkinter as tk
from tkinter import ttk

## Plots
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

#### for SCHISM
import sys
import os
#sys.path.insert(0,'/work/gg0028/SCHISM/schism-hereon-utilities/')
from schism import*

#cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable # allow axis adjustment of colorbar	  

# selection widdgets
cwd=os.getcwd()


from scipy.spatial import cKDTree
import yaml

# custom colormaps
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
#########################################



# get environment variable from EDITO
#os.environ.get("SMALL_DEMO")
userdir=os.environ.get("EDITO_INFRA_OUTPUT")
########## temp test ############

# SLR
# Min Lon  Max Lon  


##### Load input parameters from yaml file ###########
# define climatic scenario
os.chdir(userdir) # assume everything is in userdir

file=open(userdir+'nbsconfig.yaml')
config = yaml.safe_load(file)

SLR=config['options']['scenario']['SLR']
weather=config['options']['scenario']['period']

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



proj='merc'  

#### #### load model grid
#loadschism=not 's' in locals()
#if loadschism:
###### Load SCHISM
#exps=['Veg_REF',  'Veg_LE', 'Veg_HE', 'Veg_max']
setupdir=userdir+'/grid/' #['./grid/'.format(expi) for expi in exps]
os.chdir(setupdir[0])
s0=schism_setup() #second instances for overlayed plotting with mask
s=schism_setup()
os.chdir(cwd)
s.lon,s.lat=np.asarray(s.lon),np.asarray(s.lat)
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
#############################################





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
    
fig,ax = plt.subplots(figsize=(5, 3), dpi=200,ncols=1, subplot_kw={'projection': ccrs.PlateCarree()})
plot_bg_map(ax,LONmin-0.01,LONmax+0.01,LATmin-0.01,LATmax+0.01)
plot_bbox(ax,bbox)
plt.tight_layout()
ax.plot









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
 
# plot Seagrass 
fig,ax = plt.subplots(figsize=(5, 3), dpi=200,ncols=1, subplot_kw={'projection': ccrs.PlateCarree()})
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
plt.savefig(figname,dpi=300)

# results
# Plot Risk Map
# Plot Wave Height
# Plot Sediment Map

s.x,s.y=np.asarray(s.x),np.asarray(s.y)

os.chdir(userdir+'/results/SCENARIO/')
files=glob.glob('*.nc')
#os.chdir('../../')
os.chdir(userdir)

files=[file for file in files if ('X' not in file) and ('Y' not in file)]
shortconfig=[file.split('_')[0] for file in files]
varlist=shortconfig

loaddict={}
for key,val in zip(shortconfig,files):
    loaddict[key]=val

plotvars=['totalSuspendedLoad','sigWaveHeight']
units={'totalSuspendedLoad':'[g/L]','sigWaveHeight':'[m]'}

for  plotvar in plotvars:

    print('loading ' + loaddict[plotvar])

    #temp=myvar.split('_')
    # load reference scenario
    ds0=xr.open_dataset(userdir+'/results/CNTRL/'+loaddict[plotvar])
    ds1=xr.open_dataset(userdir+'/results/SCENARIO/'+loaddict[plotvar])


    myvar=list(ds1.variables)[0]
    temp=myvar.split('_')
    delta=(ds1[myvar]-ds0[myvar]).values
    plotdata=np.zeros(s.nnodes) # weight witn increasing distanes from neasrest NBS

    # interp results
    tree = cKDTree(list(zip(s.x[NBSnodes],s.y[NBSnodes])))
    qinds=np.where(plotnodes)[0]
    dists,nn=tree.query(list(zip(s.x[qinds],s.y[qinds])))
    #w=np.exp(-dists/1000)
    w=np.exp(-dists/100)
    plotdata[plotnodes]=delta[plotnodes]*w # weight

    plt.close('all')
    ph,ch,ax=s.plotAtnodesGeo(plotdata,proj='Carree',cmap=plt.cm.RdBu_r) #,ax=ax
    ax.set_title('Seagrass effect on '+ temp[0])
    #label='\Delta '+' '.join((temp[1],temp[0]))
    unit=units[plotvar]
    label=temp[0] + ' change ' + unit
    vmax=np.nanquantile(np.abs(plotdata[NBSnodes]),0.95)
    ph.set_clim(-vmax,vmax)
    ch.set_label(label)
    ax.set_extent((LONmin-0.01,LONmax+0.01,LATmin-0.01,LATmax+0.01))
    plt.tight_layout()
    figname='{:.01f}E_{:.01f}E_{:.0f}N_{:.1f}_SG{:.1f}-{:.1f}_{:s}.png'.format(LONmin,LONmax,LATmin,LATmax,Dupper,Dlower,temp[0])
    plt.savefig(figname,dpi=300)


## Risk maps
plotvar='CriticalBottomStressRatio.nc'
ds0=xr.open_dataset(userdir+'/results/CNTRL/'+loaddict[plotvar])
ds1=xr.open_dataset(userdir+'/results/SCENARIO/'+loaddict[plotvar])


plotdata[plotnodes]=delta[plotnodes]*w # weight

R1=ds0['CriticalBottomStressRatio'].values

#interpolated for measure
R2=R1.copy()
R2[plotnodes]=ds1['CriticalBottomStressRatio'].values[plotnodes]*w+R1[plotnodes]*(1-w)



from matplotlib.colors import ListedColormap, LinearSegmentedColormap
green=np.array([0/256,128/256,0/256,1])
yellow=np.array([255/256,255/256,128/256,1])
orange=np.array([255/256,165/256,0/256,1])
red=np.array([256/256,0/256,0/256,1])
clrs=np.vstack((green,yellow,orange,red))
mymap = ListedColormap(clrs)  # reguster colormap for matplotlib



# interpolation


zoom_extend=(LONmin,LONmax, LATmin,LATmax)

# default
plt.close()
ph,ch,ax=s.plotAtnodesGeo(R1,cmap=mymap,proj='Carree',region_limit=zoom_extend,extend='neither') #,extend='None'
ph.set_clim((0,1)) #
ch.set_ticks(np.linspace(0,.75,4))
ch.set_ticklabels(['No','low','increased','high'])
plt.sca(ax)
plt.title('Erosion Risk - Reference')
plt.tight_layout()
figname='{:.01f}E_{:.01f}E_{:.0f}N_{:.1f}_SG{:.1f}-{:.1f}_{:s}_default_ErosionRisk.png'.format(LONmin,LONmax,LATmin,LATmax,Dupper,Dlower,temp[0])
plt.savefig(figname,dpi=300)

# NBS
plt.close()
ph,ch,ax=s.plotAtnodesGeo(R2,cmap=mymap,proj='Carree',region_limit=zoom_extend,extend='neither') 
ph.set_clim((0,1)) #
ch.set_ticks(np.linspace(0,.75,4))
ch.set_ticklabels(['No','low','increased','high'])
plt.sca(ax)
plt.title('Erosion Risk - With NbS')
plt.tight_layout()
figname='{:.01f}E_{:.01f}E_{:.0f}N_{:.1f}_SG{:.1f}-{:.1f}_{:s}_nbs_ErosionRisk.png'.format(LONmin,LONmax,LATmin,LATmax,Dupper,Dlower,temp[0])
plt.savefig(userdir+'/'+figname,dpi=300)

## bin into intervalls for risk level reduction
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

# Example usage:
#base_cmap = 'jet'  # Existing colormap
base_cmap = 'seismic'  # Existing colormap
num_segments = 7  # Number of segments for the new colormap

segmented_cmap = create_segmented_colormap(base_cmap, num_segments)

plt.figure()
new_cmap=segmented_cmap 
ph,ch,ax=s.plotAtnodesGeo(R2bins-R1bins,proj='Carree',cmap=new_cmap,region_limit=zoom_extend) #,extend='None'
ph.set_clim((-3,3))
plt.sca(ax)
plt.title('Erosion Risk level change due to NBS')
plt.tight_layout()
figname='{:.01f}E_{:.01f}E_{:.0f}N_{:.1f}_SG{:.1f}-{:.1f}_{:s}_nbs_ErosionRiskReduction.png'.format(LONmin,LONmax,LATmin,LATmax,Dupper,Dlower,temp[0])
plt.savefig(userdir+'/'+figname,dpi=300)