#Run this from the ML folder (which should contain fake_0.txt ->-> fake_9999.txt, antpos_ful.dat and D40000-Z45-1003014.inp). Probably don't put this in Dropbox because it's going to make a big 4.32Gb numpy file and probably a movie too :)

#%matplotlib inline

from astropy.io import fits
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tqdm
from matplotlib import animation, rc
from IPython.display import HTML



wholearray='./MLtoy_antenna.list'

simdir="./test_nu/000001/"
inpfile=simdir+'000001.inp'




#### Make array flat square
#Ideally we want the array to be a flat square (or something similar) at least to start with. The first 180 antennas are in this form but the rest are randomly placed.

array_positions = np.loadtxt(wholearray)

#First, let's look at where the arrays are located in 3D.

ax = plt.figure(figsize = (10., 9.)).add_subplot(111, projection = '3d')
ax.scatter(array_positions[:, 0], array_positions[:, 1], array_positions[:, 2], s = 1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(elev = 50., azim = 220.)
plt.show()


# sorting
xs = sorted(list(set(array_positions[: 1225, 0])))
ys = sorted(list(set(array_positions[: 1225, 1])))
zs = sorted(list(set(array_positions[: 1225, 2])))
# laziness
new_array_positions = array_positions

##### Label the antennas by array position
##We want to make an array which represents the positions of each antenna in real space. This means we need the array index of each antenna so we know which array index to populate with which file. We will start at the lowest, closest antenna (in x, y and z) and go across in x until we reach the end of the array and then move along in y. We search for the antenna by position and then we add a tuple where the first element is the file/antenna label and a second element is another tuple with x and y positions

array_indices_array = []
for y_index in range(len(ys)):
    for x_index in range(len(xs)):
        antenna_location = np.array([xs[x_index], ys[y_index], zs[x_index]])
        array_indices_array.append((np.argwhere(np.equal(antenna_location, new_array_positions).all(1))[0][0], (x_index, y_index)))
array_indices = dict(array_indices_array)   
#print array_indices_array




### Anne: after producing the fake-files, all the index of the antenna in the list will give you the corresponding file index=0 -> fake0


#### Load event information
#We can load the event information from the .inp file. These will all be saved in the same event file at the end.

with open(inpfile, 'r') as file:
    for line in file.readlines():
        if 'PrimaryParticle' in line:
            particle = line.split()[1]
        if 'PrimaryEnergy' in line:
            energy = np.float(line.split()[1])
        if 'PrimaryZenAngle' in line:
            zenith = np.float(line.split()[1])
        if 'PrimaryAzimAngle' in line:
            azimuth = np.float(line.split()[1])
            
            
### Get time array
#Grab the time array from the first antenna. Since the time arrays are the same in all the files we can just use the first one. We don't actually use it at all, but it's probably worth saving anyway.

antenna_data = np.loadtxt(simdir+'/fake_0.txt')
time = antenna_data[:, 0]

### Create numpy array of entire event
#We can contain the entire event in a numpy array which is the shape of the array at each time step in the three directions. This is 4.32Gb in size, so quite big - but greatly compressed from the separate txt format and very quick to access all of the information that we might want to use.

array = np.zeros((time.shape[0], len(xs), len(ys), 3))

### Populate the array
##Simply loop through the files and put all the data from each antenna into the correct array position using the indices we found before. This might take quite a long time due to the I/O time of loading and processing the .txt file

for antenna in tqdm.tqdm(range(array_positions.shape[0])):
    antenna_data = np.loadtxt(simdir+'/fake_' + str(antenna) + '.txt')[:, 1:]
    array[:, array_indices[antenna][0], array_indices[antenna][1], :] = antenna_data

### Save the event
#We can use a .npz file format to save the data. This is a compressed binary format which can save several arrays into one file.

np.savez('event.npz', particle = particle, energy = energy, zenith = zenith, azimuth = azimuth, array = array, time = time, array_positions = new_array_positions)

### Notes
#It's important that all the array positions are constant over every single event (which is obvious) but it's also important that the time arrays are the same in every event. The initial time does not need to be the beginning of the shower. One interesting thing that we will find that if we use the initial time as the start of the shower then the network will actually extract more information (artificially) about how long it takes for the shower to reach the array (which is a priori unknown, but is available information in an online setting) to do the classification. This is either deceptive or cool depending on which way you look at it.
#Anne: the initial time when the shower starts is just know in simulations since there the time starts with the first interaction of the shower. In the expereiment, just the realtive timing between single antennas will be known.

        
        
        
        
        
#### Test loaded array
#To check that the arrays have been loaded correctly we can plot the difference between the maximum and the minimum signals over all time bins for the 100x100 array which should give a ring over the simulated antennas which are in the low-x low-y corner of the entire array.
TEST=True
if TEST:
    maximum_minimum_difference = np.max(array, axis = 0) - np.min(array, axis = 0)
    min_diff = np.min(maximum_minimum_difference)
    max_diff = np.max(maximum_minimum_difference)
    fig, ax = plt.subplots(1, 3, sharey = True, figsize = (15., 5.))
    ax[0].contourf(xs, ys, maximum_minimum_difference[:, :, 0], levels = np.linspace(min_diff, max_diff, 25))
    ax[1].contourf(xs, ys, maximum_minimum_difference[:, :, 1], levels = np.linspace(min_diff, max_diff, 25))
    ax[2].contourf(xs, ys, maximum_minimum_difference[:, :, 2], levels = np.linspace(min_diff, max_diff, 25))
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title('x voltage')
    ax[1].set_xlabel('x')
    ax[1].set_title('y voltage')
    ax[2].set_xlabel('x')
    ax[2].set_title('z voltage');

    min_array = np.min(array)
    max_array = np.max(array)
    contour = []

    rc('animation', html='html5')
    rc('figure', max_open_warning = 20000)
    fig, ax = plt.subplots(1, 3, sharey = True, figsize = (15., 5.))
    ax[0].set_xlim([xs[0], xs[-1]])
    ax[0].set_ylim([ys[0], ys[-1]])
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title('x voltage')
    ax[1].set_xlim([xs[0], xs[-1]])
    ax[1].set_ylim([ys[0], ys[-1]])
    ax[1].set_xlabel('x')
    ax[1].set_title('y voltage')
    ax[2].set_xlim([xs[0], xs[-1]])
    ax[2].set_ylim([ys[0], ys[-1]])
    ax[2].set_xlabel('x')
    ax[2].set_title('z voltage')

    images = []
    for timestep in tqdm.tqdm(range(time.shape[0])):
        image1 = ax[0].imshow(array[timestep, :, :, 0], extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect='auto', vmin = min_array, vmax = max_array)
        image2 = ax[1].imshow(array[timestep, :, :, 1], extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect='auto', vmin = min_array, vmax = max_array)
        image3 = ax[2].imshow(array[timestep, :, :, 2], extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect='auto', vmin = min_array, vmax = max_array)
        image4 = ax[0].text(41000, 21000, 'time = ' + '%.9f' % time[timestep] + 's')
        add_arts = [image1, image2, image3, image4]
        images.append(add_arts)
    ani = animation.ArtistAnimation(fig, images)
    FFwriter = animation.FFMpegWriter(fps = 100)
    ani.save('event.mp4', writer = FFwriter)
    ani