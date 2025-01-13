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


plot_or_not=False

import os, sys, string, pdb
print ( sys.argv[1:])
por=sys.argv[1:][0]
por2=sys.argv[1:][1]
print("RRRRR Arg   =",sys.argv)
print(por)
por=float(por)
str1 = ""
print(por*10)
   
##ESTE del es para borrar el valor de entrada porque si se queda WARP NO CORRE 
del sys.argv[1:]

import warp as wp
params=por
#print(f"El argumento proporcionado es: {args.numero} y su tipo es {type(args.numero)}")
zonas = 6


if True :

         # Set up solenoid lattice
    run_length = 0.2  # the lenght of the simulation 
    drift_length = 0.025 #space betweeen two sols
    solenoid_length = 0.025 # the lenght of the solenoid 
    solenoid_radius = 2.5e-2 # solenoid radius 
    NParticles = 5000
    n_grid = 150
    var1=params
    mag_solenoid=1*float(var1)
    print("TODO ESTAS AQUI XXXXXXX ",mag_solenoid)


    # prevent gist from starting upon setup
    wp.top.lprntpara = False
    wp.top.lpsplots = False
    # Define some initial BEAM  variables
    e_kin = 30.0 * wp.keV    # kinetic energy [eV]
    emit = 40.00e-6        # rms edge emittance [m-rad]
    i_beam = 10.50 * wp.mA     # beam current per species [A]
    r_x = 0.10* wp.mm       # beam edge in x [m]
    r_y = 0.10*wp.mm       # beam edge in y [m]
    r_xp = 200.0e-3        # initial beam divergence [rad]
    r_yp = r_xp           # initial beam divergence [rad]
    #wp.top.vbeam=e_kin
    #wp.top.ibeam=i_beam
    mu, sigma = 0, r_x # mean and standard deviation


    #Definition and derivation of kinematics
    moc2=0.911e9
    clight=299792458
    gammar=1+(e_kin)/moc2            # Relativistic gamma
    betar=np.sqrt(1-1.0/(gammar**2))   # Relativistic beta
    bg=betar*gammar
    velocity=betar*clight
    I0=moc2/(30) # corriente caracteristica

    #Parametro de budjer los define esto vb=I/(I0*betar)
    factor_perveance=1/(bg*bg*gammar*moc2)
    #The solenoid strenght is defined here 
    
    #mag_solenoid=0.075
    f_pr=(1.67e-27)*3e8/1.6e-19
    f_pr=(9.1e-31)*3e8/1.6e-19
    Brho=bg*f_pr
    ksol=(0.5*mag_solenoid/Brho)*(0.5*mag_solenoid/Brho)
    cgmplotfreq = 1000
    e_rms=0
    
    # Define ion species
    #protons = wp.Species(type=wp.Proton, charge_state = +1, name = "Protons")
    electrons = wp.Species(type=wp.Electron,charge_state = -1, name = "Electrons")
    
   
    #wp.derivqty()
    
    ## ASSIGNMENT for LEBT Task 1: Uncomment H2+ above and add it to the list below
    #beam_species = [electrons,hy2plus]
    beam_species = [electrons]
    for beam in beam_species:
        beam.ekin     = e_kin       # kinetic energy of beam particle [eV]
        beam.vbeam    = 0.0         # beam axial velocity [m/sec]
        beam.ibeam    = i_beam      # beam current [A]
                                
        beam.emitx    = emit        # beam x-emittance, rms edge [m-rad]
        beam.emity    = emit        # beam y-emittance, rms edge [m-rad]
        beam.vthz     = 0.0         # axial velocity spread [m/sec]
        beam.x0  = 0.0   # initial x-centroid xc = <x> [m]
        beam.y0  = 0.0   # initial y-centroid yc = <y> [m]
        beam.xp0 = 0.0   # initial x-centroid angle xc' = <x'> = d<x>/ds [rad]
        beam.yp0 = 0.0   # initial y-centroid angle yc' = <y'> = d<y>/ds [rad]
        beam.a0  = r_x   # initial x-envelope edge a = 2*sqrt(<(x-xc)^2>) [m]
        beam.b0  = r_y   # initial y-envelope edge b = 2*sqrt(<(y-yc)^2>) [m]
        beam.ap0 = r_xp  # initial x-envelope angle ap = a' = d a/ds [rad]
        beam.bp0 = r_yp  # initial y-envelope angle bp = b' = d b/ds [rad]
        
    wp.derivqty()


    for js1 in range(wp.top.pgroup.ns):
        print("particulas",js1)
    print(velocity)
    #input()
    #DEFINE DISTRIBUTION OF PARTICLES 
    XX=np.random.normal(mu, sigma, NParticles)
    YY=np.random.normal(mu, sigma, NParticles)
    radios = np.sqrt(XX**2 + YY**2)

    # Ordenar x, y y radios usando los índices
    indices_ordenados = np.argsort(radios)
    x_ordenado = XX[indices_ordenados]
    y_ordenado = YY[indices_ordenados]
    radios_ordenados = radios[indices_ordenados]
    XX=x_ordenado
    YY=y_ordenado

    ZZ=np.zeros(NParticles)
    VX=np.zeros(NParticles)
    VY=np.zeros(NParticles)
    VX=np.random.normal(mu, 10000, NParticles)
    VY=np.random.normal(mu, 10000, NParticles)

    #VZ=velocity
    sigmavz = 0.01 # mean and standard deviation
    VZ = np.random.normal(velocity, sigmavz, NParticles)
    print(VZ)



    print(np.std(XX),np.std(YY))
    if (plot_or_not):
      

        # Dividir las partículas en zonas según su índice
        zonas_indices = np.array_split(np.arange(NParticles), zonas)

        # Colores para las zonas
        colores = plt.cm.viridis(np.linspace(0, 1, zonas))

        # Graficar las partículas por zona
        plt.figure(figsize=(10, 8))
        for i, indices in enumerate(zonas_indices):
            plt.scatter(
                x_ordenado[indices],
                y_ordenado[indices],
                color=colores[i],
                label=f'Zona {i + 1}',
                s=1
            )


        plt.show()

    
    
    #wp.top.npmax = NParticles
    #wp.addparticles(x=XX,y=YY,z=ZZ,vx=VX,vy=VY,vz=VZ,js=0,lallindomain=True)
    #electrons.addparticles(x=XX,y=YY,z=ZZ,vx=VX,vy=VY,vz=VZ,lallindomain=True)#,w=3000)
    #electrons.ekin= e_kin
# SOLENOID DEFINITION
    fsol=1/(ksol*solenoid_length)
    #print("focal point lenght = ",fsol)
    ##GTY=input()
    x1=electrons.getx()
    #print("JJJ",x1)
    solenoid_zi = [drift_length + i * solenoid_length + i * drift_length for i in range(3)]
    solenoid_ze = [drift_length + (i + 1) * solenoid_length + i * drift_length for i in range(3)]
    #print("LLLLLLLLLLLLLLLLLLLLinicio final",solenoid_zi[0],solenoid_ze[0])
    wp.addnewsolenoid(zi=solenoid_zi[0],
                zf=solenoid_ze[0],
                ri=solenoid_radius,
                maxbz=0.07)  # 0.115 T for p+, 0.17 T for H2+

    wp.addnewsolenoid(zi=solenoid_zi[1],
                zf=solenoid_ze[1],
                ri=solenoid_radius,
                maxbz=0.0511)  # 0.075 T for p+, 0.125 T for H2+
    

    wp.addnewsolenoid(zi=solenoid_zi[2],
                zf=solenoid_ze[2],
                ri=solenoid_radius,
                maxbz=mag_solenoid)  # 0.075 T for p+, 0.125 T for H2+

    
    # Pipe in the solenoid transport
    pipe = wp.ZCylinderOut(radius=solenoid_radius, zlower=0.0, zupper=run_length)
    conductors=pipe

    # set up particle termination at cylindrical wall
    wp.top.prwall = solenoid_radius

  
    var1=float(params)  # number of grid cells in x and y direction
    wp.w3d.nx = n_grid
    wp.w3d.ny = n_grid

    wp.w3d.xmmax =  solenoid_radius  # x-grid max limit [m]
    wp.w3d.xmmin = -solenoid_radius  # x-grid min limit [m]
    wp.w3d.ymmax =  solenoid_radius  # y-grid max limit [m]
    wp.w3d.ymmin = -solenoid_radius  # y-grid min limit [m]
    #if mag_solenoid>0:
      #NParticles=0
    # Particle distribution options
    wp.top.npmax = NParticles
    wp.w3d.distrbtn = "KV"          # initial KV distribution

    # Random number options to use in loading
    wp.w3d.xrandom  = "digitrev"    # load x,y,z  with digitreverse random numbers
    wp.w3d.vtrandom = "digitrev"    # load vx, vy with digitreverse random numbers
    wp.w3d.vzrandom = "digitrev"    # load vz     with digitreverse random numbers
    wp.w3d.cylinder = True          # load a cylinder


    #x1=electrons.getx()
    #print("JJJ2 ",x1)
    wp.top.lrelativ   =  False    # turn off relativistic kinematics
    wp.top.relativity = 0         # turn off relativistic self-field correction
    wp.wxy.ds = 0.50e-3            # ds for part adv [m]
    wp.wxy.lvzchang = True        # Use iterative stepping, which is needed if the vz changes
    wp.top.ibpush   = 2           # magnetic field particle push: 0 - off, 1 - fast, 2 - accurate 

    # Setup field solver using 2d multigrid field solver.
    wp.w3d.boundxy = 0              # Neuman boundary conditions on edge of grid.
    wp.w3d.solvergeom = wp.w3d.XYgeom  # fieldsolve type to 2d multigrid 

    # Generate the xy PIC code.  In the generate, particles are allocated and
    # loaded consistent with initial conditions and load parameters
    # set previously.  Particles are advanced with the step() command later
    # after various diagnostics are setup.
    wp.package("wxy")    
   # wp.top.vthperp = 1.0
   # wp.top.vthz = 1.0
    wp.top.zbeam=0
    wp.generate()

    for varu in range(len(electrons.getx())):
        electrons.getx()[varu]=XX[varu]
        electrons.gety()[varu]=YY[varu]
        #electrons.getvx()[varu]=VX[varu]
        #electrons.getvy()[varu]=VY[varu]


    #wp.derivqty()
    #electrons.addparticles(x=XX,y=YY,z=ZZ,vx=VX,vy=VY,vz=VZ,lallindomain=True)#,w=3000)

    # Install conducting aperture on mesh
    wp.installconductors(conductors, dfill=wp.largepos)
    # Carry out explicit fieldsolve after generate to include conducing pipe
    # with initial beam
    wp.fieldsolve()

    # Some local runtime variables
    nsteps = int(np.ceil(run_length/wp.wxy.ds))

    #VARIABLES A GUARDAR 
    z_posi2 = []
    x_rms2  = []
    zpos = 0.0
    xiy=[]
    z_posi = []
    x_rms  = []
    x_emit = []
    rhoprome= []
    Npart_time=[]
    chi=[]
    p1x=[]
    p1xp=[]
    p1y=[]
    p1vx=[]
    p1vy=[]
    phasead=[]
    p1b=[]
    trans=[]
    ## REMOVER PARTICULA NECESARIA PARA QUE WARP GENERA EL HAZ     
    #electrons.gaminv[NParticles] = 0.
    #electrons.gaminv[0] = 0.

    
    #DEFINIR PARTIULA DE PRUEBA
    #wp.getx()[10]=0.005
    #wp.gety()[10]=0.00
    #wp.getvx()[10]=0.0
    #wp.getvy()[10]=0.0


    for i in range(nsteps):
     wp.step()
     if i % 10==0:
        print("Paso= ",i," ",int(100*(i/nsteps)),"%"," N ",len(electrons.getx()))
     #salvar DATOS PARA GUARDAR CADA PASO
     
     if i==nsteps-1:
        
        x_rms2.append(np.std(electrons.getx()))
        z_posi2.append(i*wp.wxy.ds*1e3)
        sizex=(np.std(electrons.getx()))
        e_rms=calc_emit_rms(electrons.getx(),electrons.getxp())
        Nfinal= len(electrons.getx())
        print("salida",sizex,e_rms)
    
    
    
    #input()
    f11 = open('output.txt', 'a')
    f11.write("%.4f\t%.3e\t%.3e\t%.3e" % (var1, sizex,e_rms,Nfinal) )
    f11.write("\n" )
    f11.close()

    
