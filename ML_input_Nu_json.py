#!/usr/bin/env python


''' call that script via: python createinput_fortestRM *json 
    It will read in the information in the json-file, event-by-event- and convert it to the nessary format of input for starting Zhaires sims to cross-check output of RM.
    One can distinguish/test subshowers or leading-particle showers by setting subshower=0/1
''' 



import os
from os.path import split, join, realpath
import sys
import numpy as np
import shutil
#import time
import random
import pylab as pl

import StringIO
import random
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D


# Expand the PYTHONPATH and import the radiomorphing package #NOTE: this would be on the shared disc
root_dir = realpath(join(split(__file__)[0], ".")) # = $PROJECT
sys.path.append(join(root_dir, "lib", "python"))

import retro

EARTH_RADIUS=6370949. #m
GEOMAGNET = (56.5, 63.18, 2.72) # Geomagnetic field (Amplitude [uT], inclination [deg], declination [deg]).
GdAlt=1500. 


'''NOTE: to be set to 1 (=create subshower) or 0 (=shower by leading particle)'''
subshowers=1. # subshowers introduced for each decayproduct, subshower=0. leading particle gets all the energy




##########################################################################################################
def generate_input(task=0,energy=None, azimuth=None, zenith=None, products=None, height=None, antennas=None, info=None):
    """Generate the input stream for ZHAireS."""

    zen,azim = GRANDtoZHAireS(zenith,azimuth)

    a=" ".join(map(str, products))
    b="".join( c for c in a if  c not in "(),[]''")

    seed = random.uniform(0.,1.)

    # Format the stream.
    stream = [
        "# {:s}".format(info),
        "AddSpecialParticle      RASPASSProton    /home/renault/zhaires/RASPASSprimary/RASPASSprimary Proton",
        "AddSpecialParticle      RASPASSIron      /home/renault/zhaires/RASPASSprimary/RASPASSprimary Iron",
        "AddSpecialParticle      RASPASSelectron  /home/renault/zhaires/RASPASSprimary/RASPASSprimary Electron",
        "AddSpecialParticle      RASPASSpi+  /home/renault/zhaires/RASPASSprimary/RASPASSprimary pi+",
        "AddSpecialParticle      RASPASSpi-  /home/renault/zhaires/RASPASSprimary/RASPASSprimary pi-",
        "AddSpecialParticle      RASPASSpi0  /home/renault/zhaires/RASPASSprimary/RASPASSprimary pi0",
        "AddSpecialParticle      RASPASSMulti /home/renault/zhaires/RASPASSprimary/RASPASSprimary {:s}".format(b),
        "#########################",
        "TaskName {:06d}".format(task),
        "PrimaryParticle RASPASSMulti",
        "PrimaryEnergy {:.5E} EeV".format(energy),
        "PrimaryZenAngle {:.5f} deg".format(zen),
        "PrimaryAzimAngle {:.5f} deg Magnetic".format(azim),
        "ForceModelName SIBYLL",
        "SetGlobal RASPASSHeight {:.5f} m".format(height),
        "RandomSeed {:.5f}".format(seed),
        "########################",
        "PropagatePrimary On",
        "SetGlobal RASPASSTimeShift 0.0",
        "SetGlobal RASPASSDistance 0.00"
    ]

    for a in antennas:    
        stream.append("AddAntenna {:1.2f} {:1.2f} {:1.2f}".format(a[0],a[1],a[2]))

    stream += [
        "##########################",
        "TotalShowers 1",
        "RunsPerProcess Infinite",
        "ShowersPerRun 1",
        "Atmosphere 1",
        "AddSite Ulastai 42.55 deg 86.68 deg {:.3f} m".format(1500.),
        "Site Ulastai",
        "Date 1985 10 26",
        "GeomagneticField On",
        "GeomagneticField {:.4f} uT {:.2f} deg {:.2f} deg".format(*GEOMAGNET),
        "GroundAltitude {:.3f} m".format(0.),
        "ObservingLevels 510 50 g/cm2   900 g/cm2",
        "PerShowerData Full",
        "SaveNotInFile lgtpcles All",
        "SaveNotInFile grdpcles All",
        "RLimsFile grdpcles 0.000001 m 10 km",
        "ResamplingRatio 100",
        "#########################",
        "RLimsTables 10 m 10 km",
        "ELimsTables 2 MeV 1 TeV",
        "ExportTables 5501 Opt a",
        "ExportTables 1293 Opt a",
        "ExportTables 1293 Opt as",
        "ExportTable 1205 Opt a",
        "ExportTable 1205 Opt as",
        "ExportTable 1793 Opt a",
        "ExportTable 1793 Opt as",
        "########################",
        "ForceLowEDecay Never",
        "ForceLowEAnnihilation Never",
        "########################",
        "ZHAireS On",
        "FresnelTime On",
        "FresnelFreq Off",
        "TimeDomainBin 1 ns",
        "AntennaTimeMin -100 ns",
        "AntennaTimeMax 500 ns", #can be extended until 3e-6s but if then there is still nothing then there must be a problem somewhere
        "######################", 
        "ElectronCutEnergy 1 MeV",
        "ElectronRoughCut 1 MeV",
        "GammaCutEnergy 1 MeV",
        "GammaRoughCut 1 MeV",
        "ThinningEnergy 1.e-4 Relative", #It can be 1e-5, 1e-6 or below. But running time inversely proportional to it.
        "ThinningWFactor 0.06"
    ]
    

    return "\n".join(stream)

##########################################################################################################
##########################################################################################################
def compute_antenna_pos(distance=None, inclin=0., step=1e3, nsidey=None,hz=None,GdAlt=0.):
    """ Generate antenna positions in a flat or inclined plane @ a given distance from decay"""
    """ Return N positions (x,y,z) in ZHAireS coordinates """

    if inclin!=0. and inclin!=90.:
        disty = step*nsidey
        distz = hz/(np.sin(np.radians(inclin)))
        nsidez = int(distz/step)
        nsidex = nsidez  
        distx = distz*np.cos(np.radians(inclin))
        #xi,yi = (distance,-0.5*disty) 
        #xf,yf = (distance+distx,0.5*disty)
        xi,yi = (distance,0.5*disty) 
        xf,yf = (distance+distx,0.5*disty)
        #print step*np.cos(np.radians(inclin)), step
        xx, yy= np.meshgrid(np.arange(xi,xf,step*np.cos(np.radians(inclin))),np.arange(yi,yf,step))
        zz=(xx-distance)*np.tan(np.radians(inclin))
        xxr = np.ravel(xx)
        yyr = np.ravel(yy+0.5*step)
        zzr = np.ravel(zz+GdAlt)
        xyz = np.array([xxr, yyr, zzr]).T
        
        #print len(xxr), len(yyr), len(zzr), len(xyz)
        #print xx
        
        #xi,yi = (distance-distx,-0.5*disty) 
        #xf,yf = (distance,+0.5*disty)
        #xx, yy= np.meshgrid(np.arange(xi,xf,step*np.cos(np.radians(inclin))),np.arange(yi,yf,step))
        #zz=(xx-distance)*np.tan(np.radians(inclin))
        #xxr = np.ravel(xx)
        #yyr = np.ravel(yy+0.5*step)
        #zzr = np.ravel(zz)+GdAlt
        #xyz2 = np.array([xxr, yyr, zzr]).T
        
        #xyz=np.append(xyz, xyz2)
        #print xyz
        
    elif inclin==0.:
        nsidex = int(np.round(250e3/step))
        xi,yi = (distance,-0.5*nsidey*step)
        xf,yf = (distance+(nsidex*step),0.5*nsidey*step)
        xx, yy= np.meshgrid(np.arange(xi,xf,step), np.arange(yi,yf,step))
        zz=xx*0.
        xxr = np.ravel(xx)
        yyr = np.ravel(yy+0.5*step)
        zzr = np.ravel(zz)+GdAlt
        xyz = np.array([xxr, yyr, zzr]).T

    elif inclin==90.:
        disty = step*nsidey
        distz = hz
        zi,yi = (0.,-0.5*disty)
        zf,yf = (hz,0.5*disty)
        zz, yy= np.meshgrid(np.arange(zi,zf,step),np.arange(yi,yf,step))
        xx= (yy*0.+distance)
        xxr = np.ravel(xx)
        yyr = np.ravel(yy+0.5*step)
        zzr = np.ravel(zz)+GdAlt
        xyz = np.array([xxr, yyr, zzr]).T
        
        array_display(xyz,'Selected antenna map')

    return xyz



##########################################################################################################
def array_display(ANTENNAS=None,datamap=None,title=None): #, HitCore=core):
    if len(ANTENNAS[:,0])!=0:
        fig1 = pl.figure(1,figsize=(5*3.13,3.8*3.13))
        binmap = ListedColormap(['white', 'black'], 'indexed')
        #dar=(np.max(ANTENNAS[:,0])-np.min(ANTENNAS[:,0]))/(np.max(ANTENNAS[:,1])-np.min(ANTENNAS[:,1]))
        #if dar==0:
            #dar=1
        xlbl='X [m]'
        ylbl='Y [m]'
        zlbl='Z [m]'

        ax = pl.gca(projection='3d')
        ax.scatter(ANTENNAS[:,0]*1.,ANTENNAS[:,1],ANTENNAS[:,2],c=datamap)
        ax.set_title(title)
        ax.view_init(25,-130)
        pl.xlabel(xlbl)
        pl.ylabel(ylbl)
        ax.set_zlabel(zlbl)
        #pl.gca().set_aspect(1,adjustable='box')
        
        #ax.scatter(HitCore[0], HitCore[1], HitCore[2], c='b', s=30)

        pl.show()
    return



##########################################################################################################
def random_array_pos(slope=0.,sep=1e3):
    """ Compute a random offset for 2 or 3 of the space dimensions depending of the slope """

    CORE = np.array([0.,0.,0.])
    CORE[1] = random.uniform(-1.,1.)*sep/2 # always need an offset on Y (perpendicular to trajectory)
    if slope!=90. and slope!=0.: # random position along slope => x and z are random and their offset is related
        CORE[0] = random.uniform(0.,1.)*sep*np.cos(np.radians(slope))
        CORE[2] = CORE[0]*np.sin(np.radians(slope))
    elif slope==0.: # z = 0 => no offset in Z required
        CORE[0] = random.uniform(0.,1.)*sep
    elif slope==90.: # x = distance => no offset in X required
        CORE[2] = random.uniform(0,1.)*sep

    return CORE

##########################################################################################################
def getCerenkovAngle(h=100e3):
    """ Compute the Cherenkov angle of the shower at the altitude of injection """

    # h in meters
    n = 1.+325.e-6*np.exp(-0.1218*h*1e-3)      # Refractive index ZHAireS (see email M. Tueros 25/11/2016)
    alphac = np.arccos(1./n)  
    return alphac


##########################################################################################################
def getCore(zen_rad=None, alpha=None,  distance=10000, GdAlt=1500., injh=2800.): # works with azimuth of shower =0deg, otherwise rotate the cetor later
    # define mountan slope as plae which is always facing the shower
    # r=r0 + l*u + m*v, with u*u=v*v=1, u perp to v
    az_rad=0.
    #zen_rad=np.pi-zen_rad
    #u=np.array([np.cos(az_rad+0.5*np.pi), np.sin(az_rad+0.5*np.pi),0.]) # always z=0, vector should be perp to shower axis = az_rad +0.5*pi
    u=np.array([0., 1.,0.]) # always z=0, vector should be perp to shower axis = az_rad +0.5*pi

    #u=u/mag(u)
    #print "u:", u
    #m=np.array([-np.sin(0.5*np.pi-alpha)*np.cos(az_rad+np.pi), -np.sin(0.5*np.pi-alpha)*np.sin(az_rad+np.pi), -np.cos(0.5*np.pi-alpha) ]) # describes the mountain slope and should be perp to u,0.5*np.pi-alpha to account for mountain slope, az_rad+np.pi because slope pointing towards inverse dir of shower 
    m=np.array([np.cos(alpha), 0., np.sin(alpha) ]) # describes the mountain slope and should be perp to u,0.5*np.pi-alpha to account for mountain slope, az_rad+np.pi because slope pointing towards inverse dir of shower 

    n=np.cross(u,m)
    b=np.array([distance, 0., GdAlt])
    #print "normal ", n, " u ", u, " m ", m, " b ", b
    
    
    ## define shower axis
    v =np.array([np.sin(zen_rad)*np.cos(az_rad), np.sin(zen_rad)*np.sin(az_rad), np.cos(zen_rad)]) # shower direction
    v=v/np.linalg.norm(v)
    d=np.array([0., 0., injh]) # height rt sealevel
    #print "shower axis ", v, " d" , d
    
    # find intersection point
    s=( np.dot(n,b)-np.dot(n,d) )/np.dot(n,v)
    
    core=d+s*v
    #if core[0]<distance:
        #print "not possble - check vectors",
    return core
    
    
    

##########################################################################################################
def reduce_antenna_array(injh=None,theta=None,azim=None,ANTENNAS=None, core=[5000.,0.,0.],DISPLAY=False): 
    """ Reduce the size of the initialized radio array to the shower geometrical footprint by computing the angle between shower and decay-point-to-antenna axes """
    """ theta = zenith in ZHAireS convention [in deg], injh = injection height [in m] """

    zenr = np.radians(theta)
    azimr= np.radians(azim)
    ANTENNAS1 = np.copy(ANTENNAS)

    # Shift antenna array with the randomized core position
    ANTENNAS1[:,0] = ANTENNAS1[:,0]+core[0]
    ANTENNAS1[:,1] = ANTENNAS1[:,1]+core[1]
    ANTENNAS1[:,2] = ANTENNAS1[:,2]+core[2]
    
    # Compute angle between shower and decay-point-to-antenna axes
    u_ant = ANTENNAS1+[0.,0.,-injh]
    u_ant = (u_ant.T/np.linalg.norm(u_ant,axis=1)).T
    #u_sh = [np.sin(zenr),0.,np.cos(zenr)]
    u_sh = [np.cos(azimr)*np.sin(zenr), np.sin(azimr)*np.sin(zenr), np.cos(zenr)]
    ant_angle = np.arccos(np.matmul(u_ant, u_sh))

    # Remove antennas of the initial array that are located outside the "footprint"
    omegar = getCerenkovAngle(injh)*2. #[in rad] # Accounting for a footprint twice larger than the Cherenkov angle
    angle_test = ant_angle<=omegar
    sel = np.where(angle_test)[0]
    ANTENNAS2 = ANTENNAS1[sel,:]

    # Remove the farthest antennas to reduce the number of antenna positions to simulate so that this number falls below 1000
    while np.shape(sel)[0]>1200:#999:
        x_ant_max = np.max(ANTENNAS2[:,0])
        antisel = np.where(ANTENNAS2[:,0]==x_ant_max)[0]
        ANTENNAS2= np.delete(ANTENNAS2,antisel,0)
        sel= np.delete(sel,antisel,0)

    # 3D Display of the radio array
    if DISPLAY:
        ant_map_i = np.zeros(np.shape(ANTENNAS1)[0])
        ant_map_i[sel]=1.
        cc = np.zeros((np.size(ant_map_i),3))
        cc[np.where(ant_map_i==0),:]=[1,1,1]
        #array_display(ANTENNAS,ant_angle,'Shower axis to decay point-antenna axis angle map')
        array_display(ANTENNAS1,cc,'Selected antenna map')

    return ANTENNAS2, sel

##########################################################################################################
def rotate_antenna_array(ANTENNAS=None,azim=0.): #, HitCore=None):
    """ For a azimuth different of 0, one need to rotate the radio array with azimuth """
    """ so that the longest side of the array is aligned with shower axis """

    azimr = np.radians(azim)
    print azim
    if azimr>2*np.pi:
        azimr = azimr-2.*np.pi
    elif azimr<0.:
        azim = azim+2.*np.pi
    ANTENNAS2 = np.copy(ANTENNAS)
    ANTENNAS2[:,0] = ANTENNAS[:,0]*np.cos(azimr)-ANTENNAS[:,1]*np.sin(azimr)
    ANTENNAS2[:,1] = ANTENNAS[:,0]*np.sin(azimr)+ANTENNAS[:,1]*np.cos(azimr)
    ANTENNAS2[:,2] = ANTENNAS[:,2]

    # 3D Display of the radio array
    if DISPLAY:
        ant_map_i = np.ones(np.shape(ANTENNAS2)[0])
        cc = np.zeros((np.size(ant_map_i),3))
        cc[np.where(ant_map_i==0),:]=[1,1,1]
        array_display(ANTENNAS2,cc,'Selected antenna map')
        #pl.scatter(HitCore[0], HitCore[1], HitCore[2], c='b', s=30)

    return ANTENNAS2




##########################################################################################################
def GRANDtoZHAireS(zen_DANTON=None, azim_DANTON=0):
    """ Convert coordinates from DANTON convention to ZHAireS convention """

    zen = 180. - zen_DANTON
    azim = azim_DANTON - 180.
    if azim>360:
        azim = azim-360.
    elif azim<0.:
        azim = azim+360.
    return [zen,azim]


##########################################################################################################
def mag(x):
    y=0
    for i in range(0,len(x)):
        y=y+float(x[i])*float(x[i])
        #print i , float(x[i])*float(x[i]), y
    return float(np.sqrt(float(y)))











#############################################################################################################################
#############################################################################################################################













#20000m to 50000m as distance 
#random.uniform(0, 1)

#Dd=20000.+ random.uniform(0,1)*30000. #m   #float(sys.argv[3]) #distance from decay point to beginning of radio array [m] at Grd-level
#print "DISTANCE ", Dd
#Dd=35000 #m 
slope=5# deg   #float(sys.argv[4]) #slope [deg]
hz=3000# m #float(sys.argv[5]) #Array maximum height [m]
sep=1000 # m    #float(sys.argv[6]) #separation between antennas [m]
    #try:
        #AZIMUTH = str(sys.argv[7]) #azimuth [deg] for Rotation of array
    #except:
        #AZIMUTH = str(0.)
Ny = int(np.round(35e3/sep)) #number of lines in Y direction


#################################################################
#### Initialize a too large radio antenna array
## should always be the same for all simulations
#ANTENNAS=[]
#ANTENNAS=compute_antenna_pos(Dd,slope,sep,Ny,hz,GdAlt) # NOTE: GdAlt: refernce height for array, in Simulations set height=0m, # modified c.t. originl module
#print len(ANTENNAS)
    
    
## mean of xposition
##print np.mean(ANTENNAS[:,0])
#file4= open('./MLtoy_antenna2.list', 'w')
#for i in np.arange(0,len(ANTENNAS.T[0])):
            #file4.write('{0:11.3f} {1:11.3f}  {2:11.3f}\n'.format(ANTENNAS[i,0]-np.mean(ANTENNAS[:,0]), ANTENNAS[i,1], ANTENNAS[i,2])) # pos in cm
#file4.close()



#stop
    
particle_list=[22.0, 11.0, -11.0, 111.0, 211.0, -211.0, 221.0] # 22:gamma, 11:e+-, 111:pi0, 211:pi+-, 211:eta
part_dic={'221.0':'eta','211.0': 'pi+', '-211.0': 'pi-','111.0': 'pi0', '22.0':'gamma', '13.0':'muon', '11.0': 'electron', '15.0':'tau', '16.0':'nu(t)', '321.0': 'K+', '-321.0': 'K-','130.0':'K0L', '310.0':'K0S','-323.0':'K*+'}



showerID=0
DISPLAY=1

work_dir="./"


json_file=str(sys.argv[1])
#print "json file : ", json_file

j=0
### MAYBE: this has to be done in a script which is one level higher and calling the example.py
from retro.event import EventIterator
for event in EventIterator(json_file):#"events-flat.json"): #json files contains a list of events which shall run on one node"
   showerID=showerID+1
   Dd=10000.+ random.uniform(0,1)*40000. #m   #float(sys.argv[3]) #distance from decay point to beginning of radio array [m] at Grd-level
   print "DISTANCE ", Dd
    
   
   #### to choose one specific event from a json file or test running on cluster
   j=j+1
   if j<10: 
    #if event["antennas"][0][2]>0:
        print "\n"
        print "Event ", str(event["tag"]), " started"
                            
                            
        ###DECAY
        decay_pos=event["tau_at_decay"][2]
        print "decay position ", decay_pos
        height=decay_pos[2]
        print "decay position: ", decay_pos
        decay_pos=decay_pos+np.array([0.,0.,EARTH_RADIUS]) # corrected for earth radius
        print "decay position after correction: ", decay_pos
        
        
        decay_altitude=event["tau_at_decay"][4][2] 
        print "decay decay_altitude: ", decay_altitude
        
        v=event["tau_at_decay"][3]# shower direction, assuming decay products strongly forward beamed  
        print "decay vector ", v 
        
        ###ANGLES theta, azimuth in deg (GRAND)
        print np.dot(v, decay_pos),  np.linalg.norm(decay_pos)
        theta = np.degrees(np.arccos(np.dot(v, decay_pos) / np.linalg.norm(decay_pos))) # zenith in GRAND conv.
        print "theta: ", theta
        #orthogonal projection of v onto flat plane to get the azimuth 
        x=np.array([1.,0.,0.]) #NS
        y=np.array([0.,1.,0.]) #EW
        proj_v= np.dot(v,x)*x + np.dot(v,y)*y
#        print proj_v
        azimuth = np.degrees(np.arccos(np.dot(proj_v, x))) # azimuth in GRAND conv., rt NORTH
        if proj_v[1]<0.: # y component of projection negativ, means azimuth >180deg
            azimuth = 360.-azimuth
        print "azimuth: ", azimuth
        
        if (azimuth >90.) and (azimuth<270.):
            print "azimuth outside field of view"
            continue
        

        ####### STUDY IMPACT OF SEVERAL DECAY PRODUCTS 
        #subshowers=1. # subshowers introduced for each decayproduct, subshower=0. leading particle gets all the energy
        
        if subshowers==0: #leading particle gets all the energy
                prefix="lead"
            
                ### ENERGY ep in EeV
                ep_array=np.zeros(len(event["decay"])-1)
                for i in range(1, len(event["decay"])): #len(event["decay"])-1 # gives you the number of decay products in event
                    
                    if float(event["decay"][i][0]) in particle_list: # just for valid particles 
                        pp=event["decay"][i][1] # momentum vector, second decay product: event["decay"][2][1] 
                        ep_array[i-1]=np.sqrt(pp[0]**2+pp[1]**2+pp[2]**2)# in GeV
                    print "particle ", str(i), "PID:",event["decay"][i][0]," energy in EeV: ", ep_array[i-1]*1e-9 #np.sqrt(pp[0]**2+pp[1]**2+pp[2]**2)* 1.e-9 
                etot= np.sum(ep_array)* 1.e-9 # GeV in EeV
                print "energy in EeV: ", etot
                
                ### PID primary - leading particle
                particle=int(np.argmax(ep_array) +1) # not forget about the inital neutrino and array start with 0, index of the primary
                PID= float(event["decay"][int(np.argmax(ep_array) +1)][0]) # the number of particle 
                el_list=[22.0, 11.0, -11.0, 111.0] #'22.0':'gamma', '11.0': 'electron', '-11':positron,  '111.0': 'pi0' - particle has to be in that list
                #if PID in el_list:
                    #primary="electron"
                #else: # pion-like
                    #primary="pion"
                #Get Zhaires name for that particle instead of number code
                #primary ='pi+'#str('pi+')
                #primary =str(PID)
                primary=list(part_dic.values())[list(part_dic.keys()).index(str(PID))]
                print primary
                
                multip=[]
                multip.append((primary, 1.0))
                
                
                print multip

        if subshowers==1: # all decay product get simulated
            prefix="sub"
            
            ep_array=[] #np.zeros(len(event["decay"])-1)
            PID_array=[] #np.zeros(len(event["decay"])-1)

            for i in range(1, len(event["decay"])): #len(event["decay"])-1 # gives you the number of decay products in event
                print i, float(event["decay"][i][0])
                if float(event["decay"][i][0]) in particle_list: # just for valid particles
                    pp=event["decay"][i][1] # momentum vector, second decay product: event["decay"][2][1] 
                    ep=np.sqrt(pp[0]**2+pp[1]**2+pp[2]**2)# in GeV
                    ep*=1.e-9 #in EeV
                    print "energy in EeV: ", ep
                    ep_array=np.append(ep_array, ep)
                    
                    ### PID primary
                    #part_dic={'221.0':'eta','211.0': 'pi+', '-211.0': 'pi-','111.0': 'pi0', '22.0':'gamma', '13.0':'muon', '11.0': 'electron', '15.0':'tau', '16.0':'nu(t)', '321.0': 'K+', '-321.0': 'K-','130.0':'K0L', '310.0':'K0S','-323.0':'K*+'}
                    PID= float(event["decay"][i][0])
                    #el_list=[22.0,11.0,-11.0, 111.0] #'22.0':'gamma', '11.0': 'electron', '-11.0' positron: '111.0': 'pi0'
                    #if PID in el_list:
                        #primary="electron"
                    #else: # pion-like
                        #primary="pion"
                    #primary=list(part_dic.values())[list(part_dic.keys()).index(str(PID))]
                    #print primary
                    PID_array=np.append(PID_array,PID)
                    print "PID taken: ", PID
                    
            etot= np.sum(np.asarray(ep_array)) # in EeV  
            print "PID_array ", PID_array, "energy_array ", ep_array
            
            
            multip=[]
            for i in range(0,len(ep_array)):
                multip.append((list(part_dic.values())[list(part_dic.keys()).index(str(PID_array[i]))],ep_array[i]/etot))
            print multip
                
        path, filename = os.path.split(json_file)
                  

        ## create a folder in $TMP for each event
        out_dir = "./"#+ str(event["tag"])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print "folder ", out_dir 
                    
                
        ###save ANTENNA POSITIONS in anpos.dat for each event
        ### antenna positions have to be corrected for the deacy positions since decay in morphing at (0,0,height) 
        ##antennas = join(out_dir, "antpos.dat") # file of list of desired antennas position -> in out_dir at $TMP
        #correction= np.array([decay_pos[0], decay_pos[1], 0.])
        #ant=np.copy(event["antennas"])
        #ant= np.delete(ant, np.s_[3:5], axis=1)
        ##np.savetxt(antennas, event["antennas"]-correction, delimiter='  ',fmt='%.1f')   # in GPS coordinates, later being read in in core.py
        #ANTENNAS3=ant-correction
        #print "antenna corrected: ", ANTENNAS3
        
        #print "attention for flat array correction: z of antenna set to 3m"
        
        ## approximation for flat100x100: ant_z=3m (see SLACK with VN)
        #for i in range(0,len(ANTENNAS3.T[1])):
                           #ANTENNAS3[i,2]=3. #m 
        
                           
###### ANTENNA POSITIONS                           
        ANTENNAS=np.genfromtxt('./MLtoy_antenna.list')
    
        ### Randomize the position of the core of the array in Y  (along EAST WEST)
        #NOTE: check what this function is doing
        ra=np.random.rand(1,1)-0.5
        r=ra[0,0]*35000./2.
        d=Dd* np.tan(np.radians(azimuth))
        print r, d
        CORE = np.array([0.,r+d,0.])#random_array_pos(slope,sep)
        print "Core at : ", CORE
        #ANTENNAS.T[0]=ANTENNAS.T[1] + CORE[1]
        ANTENNAS.T[0]=ANTENNAS.T[0] + Dd
        
        ### Reduce the radio array to the shower geometrical footprint (we account for a footprint twice larger than the Cherenkov angle)
        ANTENNAS2, sel = reduce_antenna_array(decay_altitude,theta, azimuth,ANTENNAS,CORE,DISPLAY)#, core)
        #print ANTENNAS[sel] #sel
        
        # refernce our has to be stored for comparison im treatment.py
        ANTENNAS.T[1]=ANTENNAS.T[1]+CORE[1]
       
        
        
        AZ=0. # array always facing NORTH
        if AZ!=0.:
                ANTENNAS3 = rotate_antenna_array(ANTENNAS2,azimuth)
        else:
                ANTENNAS3 = np.copy(ANTENNAS2)
                
                #ANTENNAS3 = rotate_antenna_array(ANTENNAS2,azim)
                
        # what I need is the index of that chosen antennas and write the antenna with the specific index to the inpfile. not the spoistion of thechosen antenna since then it will be alwaysthe same.
                
                
        info= "tag: "+str(event["tag"]) + ", in json:" + filename, "distance to decay in m: ", Dd, " core off in y inm ", r+d

        ### produce ZHaires file
        if ANTENNAS3.size:
            #print('array is not empty')
            print "numbers of antennas simulated: ", len(ANTENNAS3)
            
            ##################################################################
            #Output directory for ZHAireS results.
            #DATAFS = work_dir+'/showerdata/'
            #showerdata_file = DATAFS+os.path.splitext(os.path.basename(danton_lib))[0]+'-showerdata.txt'
            inp_dir = work_dir+'/' +"{:06d}".format(showerID) +'/'
            #if not(os.path.isdir(DATAFS)):
                #os.mkdir(DATAFS)
            if not(os.path.isdir(inp_dir)):
                os.mkdir(inp_dir)
                
            fileZha = inp_dir+"{:06d}".format(showerID)+ '.inp'
            dir=os.path.dirname(fileZha)
            if not os.path.exists(dir):
                os.makedirs(dir)
            
            ### Write the ZHAireS input file
            inpfile = open(fileZha,"w+")
            print fileZha
            
            totito  = generate_input(showerID, etot, azimuth, theta, multip, decay_altitude,ANTENNAS3, info) # modified c.t. originl module
            inpfile.write(totito)
            inpfile.close()
            
            ref_array=inp_dir+'/' +"MLtoy_{:06d}.list".format(showerID)
            file4= open(ref_array, 'w')
            for i in np.arange(0,len(ANTENNAS.T[0])):
                file4.write('{0:11.3f} {1:11.3f}  {2:11.3f}\n'.format(ANTENNAS[i,0], ANTENNAS[i,1], ANTENNAS[i,2])) # pos in cm
            file4.close() 
        else:
            print('array is empty - no antennas simulated')

                    
                           
  
  
        #path, filename = os.path.split(json_file)
        #info= "tag: "+str(event["tag"]) + ", in json:" + filename
  
##### Write the ZHAireS input file
        #showerID=str(prefix)+"_"+str(j)
        #fileZha=join(out_dir, "SIM_"+str(showerID)+".inp")
        #inpfile = open(fileZha,"w+")
        ##showerID=str(prefix)+"_"+str(j) # number event in file, tag as comment in .inp-file
        #totito  = generate_input(showerID, etot, azimuth, theta, multip, decay_altitude,ANTENNAS3, info)
        #inpfile.write(totito)
        #inpfile.close()    
        
    #else:
        #print "failed"
                    

print "Job done"


