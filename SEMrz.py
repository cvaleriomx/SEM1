"""
Example Pierce diode calculation.
Hot plate source emitting singly ionized potassium
"""
from curses.panel import top_panel
#from turtle import shape
#import os,sys
#import argparse
import matplotlib.pyplot as plt
import numpy as np
#from libreria import multiplot, multiplot2, calc_rms, multiplot_magnetized, calc_emit_rms
from libreria import *

import scipy.optimize as optimize
import gc

#select if plot or not
plot_or_not=True


import os, sys, string, pdb
print ( sys.argv[1:])
por=sys.argv[1:][0]
por2=sys.argv[1:][1]
por3=sys.argv[1:][2]
print("RRRRR Arg   =",sys.argv)
print(por)
por=float(por)
por2=float(por2)
por3=float(por3)
str1 = ""
print(por*10)
   
##ESTE del es para borrar el valor de entrada porque si se queda WARP NO CORRE 
del sys.argv[1:]






import warp as wp
from warp.run_modes.egun_like import gun
from libreria import *
# --- Set four-character run id, comment lines, user's name.
# wp.top.pline2 = "Pierce diode example"
# wp.top.pline1 = "Injected beam. Semi-Gaus."
# wp.top.runmaker = "DPG"
wp.top.lpsplots = plot_or_not
# --- Invoke setup routine for the plotting

if plot_or_not:
    wp.setup()

# --- Set the dimensionality
wp.w3d.solvergeom = wp.w3d.RZgeom

# --- Sets method of running
# ---   Steady state gun mode
# ---   Time dependent simulation (when False)
steady_state_gun = False

# --- Basic parameters
channel_radius = 2.56*wp.cm

diode_voltage = 93.*wp.kV

# --- Setup source plate
source_radius = 0.2*wp.cm
source_temperature = 0.1  # in eV
source_curvature_radius = 30.*wp.cm  # --- radius of curvature of emitting surface
pierce_angle = 67.

# --- Setup diode aperture plate
rplate = 5.5*wp.cm  # --- aperture radius
plate_width = 2.5*wp.cm  # --- thickness of aperture plate

# Define some initial BEAM  variables
e_kin = 30.0 * wp.keV    # kinetic energy [eV]
emit = 4.00e-6        # rms edge emittance [m-rad]
i_beam = 10.50 * wp.mA     # beam current per species [A]
r_x = 0.30* wp.mm       # beam edge in x [m]
r_y = 0.30*wp.mm       # beam edge in y [m]
r_xp = 100.0e-3        # initial beam divergence [rad]
r_yp = r_xp           # initial beam divergence [rad]
#calculate twiss parameter alpha and beta for a beam with given emittance and size
beta_x = r_x**2/emit
alpha_x = -r_x*r_xp/emit
print("beta_x", beta_x)
print("alpha_x", alpha_x)
#input()

#wp.top.vbeam=e_kin
#wp.top.ibeam=i_beam
mu, sigma = 0, r_x # mean and standard deviation


#Definition and derivation of kinematics
moc2=0.511e6
clight=299792458
gammar=1+(e_kin)/moc2            # Relativistic gamma
betar=np.sqrt(1-1.0/(gammar**2))   # Relativistic beta
bg=betar*gammar
velocity=betar*clight
I0=moc2/(30) # corriente caracteristica

#Parametro de budjer los define esto vb=I/(I0*betar)
factor_perveance=1/(bg*bg*gammar*moc2)
#The solenoid strenght is defined here 

mag_solenoid=0.075
f_pr=(1.67e-27)*3e8/1.6e-19
f_pr=(9.1e-31)*3e8/1.6e-19
Brho=bg*f_pr

ksol=(0.5*mag_solenoid/Brho)*(0.5*mag_solenoid/Brho)
cgmplotfreq = 1000
e_rms=0

# Define ion species
#protons = wp.Species(type=wp.Proton, charge_state = +1, name = "Protons")
electrons = wp.Species(type=wp.Electron,charge_state = -1, name = "Electrons")

source_temperature = 0.01  # in eV
vtrans= clight*np.sqrt(source_temperature/moc2)
#wp.derivqty()

## ASSIGNMENT for LEBT Task 1: Uncomment H2+ above and add it to the list below
    #beam_species = [electrons,hy2plus]
beam_species = [electrons]
for beam in beam_species:

        print("velocity transvesal",source_temperature*wp.jperev/beam.mass,vtrans)
        print("velocity transvesal",vtrans/velocity)
        #input()
            # --- Set basic beam parameters
        beam.a0 = r_x   # initial x-envelope edge a = 2*sqrt(<(x-xc)^2>) [m]
        beam.b0 = r_y   # initial y-envelope edge b = 2*sqrt(<(y-yc)^2>) [m]
        beam.ap0 = r_xp  # initial x-envelope angle ap = a' = d a/ds [rad]
        beam.bp0 = r_yp  # initial y-envelope angle bp = b' = d b/ds [rad]
        beam.ibeam = -0.0001 * wp.mA
        beam.vthz = vtrans
        beam.vthperp = vtrans*50
        beam.ekin = e_kin
        
        # beam.ekin     = e_kin       # kinetic energy of beam particle [eV]
        # beam.vbeam    = 0.0         # beam axial velocity [m/sec]
        # beam.ibeam    = i_beam      # beam current [A]
                                
        # beam.emitx    = emit        # beam x-emittance, rms edge [m-rad]
        # beam.emity    = emit        # beam y-emittance, rms edge [m-rad]
        # beam.vthz     = 0.0         # axial velocity spread [m/sec]
        # beam.x0  = 0.0   # initial x-centroid xc = <x> [m]
        # beam.y0  = 0.0   # initial y-centroid yc = <y> [m]
        # beam.xp0 = 0.0   # initial x-centroid angle xc' = <x'> = d<x>/ds [rad]
        # beam.yp0 = 0.0   # initial y-centroid angle yc' = <y'> = d<y>/ds [rad]
        # beam.a0  = r_x   # initial x-envelope edge a = 2*sqrt(<(x-xc)^2>) [m]
        # beam.b0  = r_y   # initial y-envelope edge b = 2*sqrt(<(y-yc)^2>) [m]
        # beam.ap0 = r_xp  # initial x-envelope angle ap = a' = d a/ds [rad]
        # beam.bp0 = r_yp  # initial y-envelope angle bp = b' = d b/ds [rad]

wp.derivqty()

# --- Preallocate fixed-size arrays for final selected particles
MAX_ACCUM_PARTICLES = 100000
accum_xsel = np.zeros(MAX_ACCUM_PARTICLES, dtype=np.float64)
accum_ysel = np.zeros(MAX_ACCUM_PARTICLES, dtype=np.float64)
accum_zsel = np.zeros(MAX_ACCUM_PARTICLES, dtype=np.float64)
accum_count = 0

# --- Length of simulation box
runlen =  35.*wp.cm

# --- Set boundary conditions
# ---   for field solve
wp.w3d.bound0 = wp.neumann
wp.w3d.boundnz = wp.neumann
wp.w3d.boundxy = wp.dirichlet
# ---   for particles
wp.top.pbound0 = wp.absorb
wp.top.pboundnz = wp.absorb
wp.top.prwall = channel_radius

# --- Set field grid size
wp.w3d.xmmin = -channel_radius
wp.w3d.xmmax = +channel_radius
wp.w3d.ymmin = -channel_radius
wp.w3d.ymmax = +channel_radius
wp.w3d.zmmin = 0.
wp.w3d.zmmax = runlen

# --- Field grid dimensions - note that nx and ny must be even.
wp.w3d.nx = wp.w3d.ny =64
wp.w3d.nz = 264

# --- Set the time step size. This needs to be small enough to satisfy the Courant limit.
dz = (wp.w3d.zmmax - wp.w3d.zmmin)/wp.w3d.nz
vzfinal =velocity

wp.top.dt = 1*(dz/vzfinal)

# --- Specify injection of the particles
wp.top.inject = 1  # 2 means space-charge limited injection
#wp.top.rinject = source_curvature_radius  # Source radius of curvature
wp.top.npinject = 2550  # Approximate number of particles injected each step
#wp.top.vinject = diode_voltage
#wp.w3d.l_inj_exact = True

# --- If using the RZ geometry, set so injection uses the same geometry
wp.w3d.l_inj_rz = (wp.w3d.solvergeom == wp.w3d.RZgeom)

# --- Set up fieldsolver
wp.f3d.mgtol = 1.e-1  # Multigrid solver convergence tolerance, in volts

solver = wp.MultiGrid2D()
wp.registersolver(solver)

piercezlen = (channel_radius - source_radius)*wp.tan((90.-pierce_angle)*wp.pi/180.)
piercezlen = 0.04
rround = plate_width/2.

# --- Create source conductors


# --- the rsrf and zsrf specify the line in RZ describing the shape of the source and Pierce cone.
# --- The first segment is an arc, the curved emitting surface.
# source = wp.ZSrfrv(rsrf=[0., source_radius, rpierce, channel_radius, channel_radius],
#                    zsrf=[0., sourcezlen, sourcezlen + piercezlen, sourcezlen + piercezlen, 0.],
#                    zc=[source_curvature_radius, None, None, None, None],
#                    rc=[0., None, None, None, None],
#                    voltage=diode_voltage)

#wp.installconductor(source, dfill=wp.largepos)
print("solenois_sem.TXT")
#realistic solenoid field from Poisson calculation
realistic_solenoid_field = False
if realistic_solenoid_field:
    filename1 = "solenois_sem.TXT"
    filename1 = "solenois_sem_normalizedr0.TXT"
    bx, by, bz, psol = cargar_campo_poisson_3d_para_warp(
        filename1,
        Nx=101,
        Ny=101,
        Nz=201
    )
    print("solenoide z pocision",psol["z_start"], psol["z_stop"],bz.max())
    #input()
    factorsol=por
    z0_sol=0.0
    # --- Add the solenoid field as a background field in Warp the solenoid center is at z=0.08 m and the length is 0.16 m, so it extends from 0.0 to 0.16 m in the simulation box
    wp.addnewbgrd(
        psol["z_start"]+z0_sol, psol["z_stop"]+z0_sol,
        xs=psol["xs"],
        dx=psol["dx"],
        ys=psol["ys"],
        dy=psol["dy"],
        nx=psol["nx"],
        ny=psol["ny"],
        nz=psol["nz"],
        bx=bx*factorsol,
        by=by*factorsol,
        bz=bz*factorsol
    )
    z0_sol2=0.08
    factorsol2=por2

    wp.addnewbgrd(
        psol["z_start"]+z0_sol2, psol["z_stop"]+z0_sol2,
        xs=psol["xs"],
        dx=psol["dx"],
        ys=psol["ys"],
        dy=psol["dy"],
        nx=psol["nx"],
        ny=psol["ny"],
        nz=psol["nz"],
        bx=bx*factorsol2,
        by=by*factorsol2,
        bz=bz*factorsol2
    )
    print("solenois_sem.TXT2")

    filename2 = "objetivo_normalized_r0.TXT"
    bx, by, bz, psol = cargar_campo_poisson_3d_para_warp(
        filename2,
        Nx=51,
        Ny=51,
        Nz=201
    )

    z0_sol_obje=-0.04+0.3
    factorsol3=por3
    wp.addnewbgrd(
        psol["z_start"]+z0_sol_obje, psol["z_stop"]+z0_sol_obje,
        xs=psol["xs"],
        dx=psol["dx"],
        ys=psol["ys"],
        dy=psol["dy"],
        nx=psol["nx"],
        ny=psol["ny"],
        nz=psol["nz"],
        bx=bx*factorsol3,
        by=by*factorsol3,
        bz=bz*factorsol3
    )
    #To be sure plot the solenoid field
    if plot_or_not:
        #create surface plot of bz in the plane z,x
        print(psol["z_start"]+z0_sol, psol["z_stop"]+z0_sol, psol["nz"])
        z = np.linspace(psol["z_start"]+z0_sol, psol["z_stop"]+z0_sol, psol["nz"])
        x = np.linspace(psol["xs"], psol["xs"] + psol["nx"]*psol["dx"], psol["nx"])
        X, Z = np.meshgrid(x, z)
        plt.figure(figsize=(8, 6))
        #check the shape of bz and Z and X

        print("bz shape", bz.shape)
        print("Z shape", Z.shape)
        print("X shape", X.shape)
        plt.contourf(Z, X, bz[psol["ny"]//2, :, :].T*factorsol, levels=50, cmap='viridis')
        plt.colorbar(label='Bz (T)')
    
        plt.show()
else:
    # --- Add a simple hard-edge solenoid field as a background field in Warp the solenoid center is at z=0.08 m and the length is 0.16 m, so it extends from 0.0 to 0.16 m in the simulation box
   z0_sol_center=0.08

   z1_sol_center=0.16
   z2_sol_center=0.3
   length_sol=25*wp.mm
   length_objetivo=8*wp.mm
   solenoid_radius=0.02
   solenoid_radius_objetivo=0.01
   wp.addnewsolenoid(zi=z0_sol_center-length_sol/2, zf=z0_sol_center+length_sol/2, ri=solenoid_radius, maxbz=por)
   wp.addnewsolenoid(zi=z1_sol_center-length_sol/2, zf=z1_sol_center+length_sol/2, ri=solenoid_radius, maxbz=por2)
   wp.addnewsolenoid(zi=z2_sol_center-length_objetivo/2, zf=z2_sol_center+length_objetivo/2, ri=solenoid_radius_objetivo, maxbz=por3)









# --- Create aperture plate with inner radius 5 mm, thickness 3 mm, centered at z = 0.12 m
plate_aperture_radius = 0.0005
plate_thickness = 0.003
plate_zcenter = 0.12
plate_zlower = plate_zcenter - plate_thickness/2
plate_zupper = plate_zcenter + plate_thickness/2
rad_val = max(plate_aperture_radius, plate_thickness)
# aperture_plate = wp.ZSrfrv(
#      rsrf=[plate_aperture_radius, plate_aperture_radius, channel_radius*0.9, channel_radius*0.9],
#      zsrf=[plate_zlower, plate_zupper, plate_zupper, plate_zlower],
#      #rad=[rad_val,None, None, None],
#      rc=[plate_aperture_radius-plate_thickness, None, None, None],
#      zc=[plate_zcenter, None, None, None],
#      voltage=0
#  )
aperture_plate = wp.ZRoundedCylinderOut(radius=plate_aperture_radius, length=plate_thickness, radius2=plate_thickness/2, voltage=0., zcent=plate_zcenter)
#secon aperture after the solenoid
plate_zcenter2 = 0.25
plate_aperture_radius2 = 0.0005
aperture_plate2 = wp.ZRoundedCylinderOut(radius=plate_aperture_radius2, length=plate_thickness, radius2=plate_thickness/2, voltage=0., zcent=plate_zcenter2)
plate_zcenter3 = 0.3
plate_aperture_radius3 = 0.001
aperture_plate3 = wp.ZRoundedCylinderOut(radius=plate_aperture_radius3, length=plate_thickness, radius2=plate_thickness/2, voltage=0., zcent=plate_zcenter3)

wp.installconductor(aperture_plate2)
wp.installconductor(aperture_plate)
wp.installconductor(aperture_plate3)

# --- Pipe in the solenoid transport
pipe = wp.ZCylinderOut(radius=0.025, zlower= 0, zupper= runlen-0.005,voltage=0)

wp.installconductor(pipe, dfill=wp.largepos)


#wp.installconductor(plate, dfill=wp.largepos)

# --- Setup the particle scraper
scraper = wp.ParticleScraper([pipe, aperture_plate, aperture_plate2, aperture_plate3])

# --- Set pline1 to include appropriate parameters
if wp.w3d.solvergeom == wp.w3d.RZgeom:
    wp.top.pline1 = ("Injected beam. Semi-Gaus. %dx%d. npinject=%d, dt=%d" %
                     (wp.w3d.nx, wp.w3d.nz, wp.top.npinject, wp.top.dt))
else:
    wp.top.pline1 = ("Injected beam. Semi-Gaus. %dx%dx%d. npinject=%d, dt=%d" %
                     (wp.w3d.nx, wp.w3d.ny, wp.w3d.nz, wp.top.npinject, wp.top.dt))

# --- Generate the PIC code (allocate storage, load ptcls, t=0 plots, etc.)
wp.package("w3d")
wp.generate()

# --- Open up plotting windows
if plot_or_not:

    wp.winon()
    wp.winon(1, suffix='current')


def beamplots():
    wp.window(0)
    wp.fma()
    wp.pfzr(plotsg=0, cond=0, titles=False)
    #wp.pcbzy(plotsg=0, cond=0, titles=False)
    pipe.draw(filled=150, fullplane=False)
    aperture_plate.draw(filled=150, fullplane=False)
    aperture_plate2.draw(filled=150, fullplane=False)
    aperture_plate3.draw(filled=150, fullplane=False)
    #plate.draw(filled=100, fullplane=False)
    wp.ppzr(titles=False)
    #wp.limits(0.3, wp.w3d.zmmaxglobal, 0., 0.1*channel_radius)
    wp.limits(0.3, wp.w3d.zmmaxglobal, 0., 0.001)

    #wp.limits(wp.w3d.zmminglobal, wp.w3d.zmmaxglobal, 0., channel_radius)
    #wp.limits(0.1, 0.13, 0., 0.007)
    wp.ptitles('Hot plate source', 'Z (m)', 'R (m)')
    wp.refresh()

    wp.window(1)
    wp.fma()
    wp.pzcurr()
    #wp.limits(wp.w3d.zmminglobal, wp.w3d.zmmaxglobal, 0., diode_current*1.5)
    wp.refresh()
    # --- Plotting the beam size at the end of the simulation box
    wp.window(2)
    wp.fma()
    zcut = runlen - 0.01
    xsel, ysel, zsel = save_final_particles(save_particle_file=False)
    wp.ppgeneric(xsel, ysel)
    wp.limits(-channel_radius, channel_radius, -channel_radius, channel_radius)
    wp.ptitles('Hot plate source', 'X (m)', 'Y (m)')
    wp.refresh()
    #if wp.top.it % 320 == 0:
    if False:
        plt.figure(figsize=(6, 5))
        plt.scatter(electrons.getz(), electrons.getbz(), s=1, alpha=0.5)
        plt.show()

def save_final_particles(it=None, save_particle_file=True):
    global accum_count, accum_xsel, accum_ysel, accum_zsel

    zcut = runlen - 0.01
    if it is None:
        it = int(getattr(wp.top, 'it', 0))

    size_history_file = 'beam_size_history.csv'
    if it % 20 == 0:
        if not os.path.exists(size_history_file):
            with open(size_history_file, 'w') as f:
                f.write('iter,species,n_particles,sigma_x,sigma_y,sigma_r\n')

    for sp in beam_species:
        try:
            x = sp.getx()
            y = sp.gety()
            z = sp.getz()
        except Exception:
            continue

        mask = (z > zcut)
        xsel = x[mask]
        ysel = y[mask]
        zsel = z[mask]
        nsel = xsel.size

        if nsel > 0:
            space_left = MAX_ACCUM_PARTICLES - accum_count
            if space_left > 0:
                add_count = min(nsel, space_left)
                accum_xsel[accum_count:accum_count + add_count] = xsel[:add_count]
                accum_ysel[accum_count:accum_count + add_count] = ysel[:add_count]
                accum_zsel[accum_count:accum_count + add_count] = zsel[:add_count]
                accum_count += add_count
                if add_count < nsel:
                    print(f'Accumulated {add_count} particles, array full at {accum_count}/{MAX_ACCUM_PARTICLES}.')
                else:
                    print(f'Accumulated {add_count} particles, total now {accum_count}.')
            else:
                print('Accumulated particle arrays are full; skipping additional particles.')

        if nsel > 0 and it % 20 == 0:
            if accum_count > 0:
                acc_x = accum_xsel[:accum_count]
                acc_y = accum_ysel[:accum_count]
                sigma_x = np.sqrt(np.mean(acc_x**2))
                sigma_y = np.sqrt(np.mean(acc_y**2))
                sigma_r = np.sqrt(np.mean((acc_x**2 + acc_y**2) / 2.0))
            else:
                sigma_x = np.sqrt(np.mean(xsel**2))
                sigma_y = np.sqrt(np.mean(ysel**2))
                sigma_r = np.sqrt(np.mean((xsel**2 + ysel**2) / 2.0))
            if save_particle_file:
                fname = 'particles_final_it{0:04d}_{1}.npz'.format(it, sp.name)
                np.savez(fname, x=xsel, y=ysel, z=zsel)
                print('Saved', fname, 'with', nsel, 'particles (z >', zcut, ')')
            with open(size_history_file, 'a') as f:
                f.write('{0},{1},{2},{3:.6e},{4:.6e},{5:.6e}\n'.format(
                    it, sp.name, nsel, sigma_x, sigma_y, sigma_r))
            print('Saved beam size for', sp.name, 'at it=', it)

    return accum_xsel[:accum_count], accum_ysel[:accum_count], accum_zsel[:accum_count]


def plot_beam_size_history(csvfile='beam_size_history.csv', show=False):
    if not os.path.exists(csvfile):
        print('No beam size history file found:', csvfile)
        return
    data = np.genfromtxt(csvfile, delimiter=',', names=True, dtype=None, encoding='utf-8')
    if data.size == 0:
        print('Beam size history file is empty:', csvfile)
        return

    species_names = np.unique(data['species'])
    plt.figure(figsize=(10, 6))
    for sp in species_names:
        spmask = data['species'] == sp
        iters = data['iter'][spmask]
        sigma_x = data['sigma_x'][spmask]
        sigma_y = data['sigma_y'][spmask]
        sigma_r = data['sigma_r'][spmask]
        plt.plot(iters, sigma_x, label=f'{sp} sigma_x')
        plt.plot(iters, sigma_y, label=f'{sp} sigma_y')
        plt.plot(iters, sigma_r, '--', label=f'{sp} sigma_r')

    plt.xlabel('Iteration')
    plt.ylabel('Beam size (m)')
    plt.title('Beam size history vs iteration')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig('beam_size_history.png')
    if show:
        plt.show()
    plt.close()


if steady_state_gun:
    # --- Steady-state operation
    # --- This does steady-state gun iterations, plotting the z versus r
    # --- after each iteration.
    wp.top.inj_param = 0.2
    for iter in range(100):
        gun(1, ipstep=1, lvariabletimestep=1)
        if wp.top.it % 20  == 0 and plot_or_not:
            beamplots()
        
        

else:

    # --- Call beamplots after every 20 steps
    @wp.callfromafterstep
    def makebeamplots():
        if wp.top.it % 20  == 0 and plot_or_not:
            beamplots()

    @wp.callfromafterstep
    def make_save_final_particles():
        if wp.top.it % 20 == 0:
            save_final_particles(save_particle_file=False)

    wp.step(1500)

# --- Make sure that last plot frames get sent to the cgm file
if plot_or_not:
    wp.window(0)
    wp.hcp()
    wp.window(1)
    wp.hcp()
# --- Plotting the beam size at the end of the simulation box create two plots one with the final particles and another with the beam size history  
#sing axia
if plot_or_not:
    fig4, axs2 = plt.subplots(2, 1, figsize=(10, 6))
    # --- Plot beam size history after the simulation
    #plot histogram 2d of final particles
    xsel, ysel, zsel = save_final_particles(save_particle_file=False)
    hxy, xedges, yedges  = np.histogram2d(xsel, ysel, bins=100)
    xhy, xhx = np.meshgrid(xedges, yedges)
    #report beam size at the end of the simulation
    sigma_x_final = np.sqrt(np.mean(xsel**2))
    sigma_y_final = np.sqrt(np.mean(ysel**2))
    sigma_r_final = np.sqrt(np.mean((xsel**2 + ysel**2) / 2.0))
    print("Final beam size", sigma_x_final, sigma_y_final, sigma_r_final)
    pcm = axs2[0].pcolormesh(xedges, yedges, hxy.T, shading='auto', cmap='viridis')
    fig4.colorbar(pcm, ax=axs2[0], label='Particle count')
    #cntr1= axs2[1].contourf(xhy, xhx, np.ma.masked_invalid(ppp),
    #                      corner_mask=True, linewidths=0.5)
    #.colorbar( label='Particle count')

    axs2[0].set_xlabel('X (m)')
    axs2[0].set_ylabel('Y (m)')
    #Now histogram 1d of beam size history
    axs2[1].hist(xsel, bins=120, color='blue', alpha=0.7)
    axs2[1].set_xlabel('X (m)')
    axs2[1].set_ylabel('Particle count')
    axs2[1].set_title('Beam size history')  

    plt.show()

plot_beam_size_history(show=plot_or_not)