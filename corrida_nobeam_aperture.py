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




plot_or_not=True


import os, sys, string, pdb
print ( sys.argv[1:])
por=sys.argv[1:][0]
print(por)
por=float(por)
por2=sys.argv[1:][1]
var2=float(por2)
mag_solenoid2=var2
str1 = ""
print(por*10)
   
##ESTE del es para borrar el valor de entrada porque si se queda WARP NO CORRE 
del sys.argv[1:]
import warp as wp
params=por
#print(f"El argumento proporcionado es: {args.numero} y su tipo es {type(args.numero)}")
zonas = 6


h = 6.626e-34  # Constante de Planck en J·s
m_e = 9.109e-31  # Masa del electrón en kg
eV_to_J = 1.602e-19  # Conversión de eV a J
#Definition and derivation of kinematics
moc2=0.911e9
clight=299792458

if True :
    

    # Energía cinética del electrón en J
    E_k_eV = 30e3  # Energía en keV
 




        # Set up solenoid lattice
    run_length = 0.15  # the lenght of the simulation 
    drift_length = 0.025 #space betweeen two sols
    solenoid_length = 0.025 # the lenght of the solenoid 
    solenoid_radius = 1.0e-2 # solenoid radius 
    NParticles = 65000
    n_grid = 150
    var1=params
    mag_solenoid=1*float(var1)
    mag_solenoid2=1*float(var1)

    print("TODO ESTAS AQUI XXXXXXX ",mag_solenoid)




    # prevent gist from starting upon setup
    wp.top.lprntpara = False
    wp.top.lpsplots = False
    # Define some initial BEAM  variables
    e_kin = 30.0 * wp.keV    # kinetic energy [eV]
    emit = 40.00e-6        # rms edge emittance [m-rad]
    i_beam = 0.00005 * wp.mA     # beam current per species [A]
    r_x = 0.10* wp.mm       # beam edge in x [m]
    r_y = 0.10*wp.mm       # beam edge in y [m]
    r_xp = 70.0e-3        # initial beam divergence [rad]
    r_yp = r_xp           # initial beam divergence [rad]
    #wp.top.vbeam=e_kin
    #wp.top.ibeam=i_beam
    mu, sigma = 0, r_x # mean and standard deviation





    gammar=1+(e_kin)/moc2            # Relativistic gamma
    betar=np.sqrt(1-1.0/(gammar**2))   # Relativistic beta
    bg=betar*gammar
    velocity=betar*clight
    I0=moc2/(30) # corriente caracteristica
    
    
   
   
    # Energía total (E_total = E_k + E_rest)
    E_total = e_kin + moc2
    # Momento del electrón: p = sqrt((E_total/c)^2 - (m_e * c)^2)
    p_ele = ((E_total / clight)**2 - (moc2)**2)**0.5
    # Longitud de onda de de Broglie: lambda = h / p
    wavelength = h / p_ele

    print("longitud onda ",wavelength)


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
    #if (plot_or_not):
    if (False):




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
    print("LLLLLLLLLLLLLLLLLLLLinicio final",solenoid_zi,solenoid_ze)
    #input()
    wp.addnewsolenoid(zi=solenoid_zi[0],
                zf=solenoid_ze[0],
                ri=solenoid_radius,
                maxbz=0.062)  # 0.062 .07


    wp.addnewsolenoid(zi=solenoid_zi[1],
                zf=solenoid_ze[1],
                ri=solenoid_radius,
                maxbz=1.0*mag_solenoid)  # 0.075 T for p+, 0.125 T for H2+
    
#0.0511  0.0539

    wp.addnewsolenoid(zi=solenoid_zi[2],
                zf=solenoid_zi[2]+0.01,
                ri=solenoid_radius*0.5,
                maxbz=1*mag_solenoid2)  # 0.075 T for p+, 0.125 T for H2+ [0.03345496 0.06552982]

#0.0575 0.06039
    
    # Pipe in the solenoid transport
    pipe = wp.ZCylinderOut(radius=solenoid_radius*1.0, zlower=0.0, zupper=solenoid_ze[0])
    pipe1 = wp.ZCylinderOut(radius=0.001, zlower=solenoid_zi[1], zupper=solenoid_zi[1]+0.005)
    pipe2 = wp.ZCylinderOut(radius=0.001, zlower=solenoid_zi[2], zupper=solenoid_zi[2]+0.004)
    pipe3 = wp.ZCylinderOut(radius=0.004, zlower=solenoid_ze[2], zupper=solenoid_ze[2]+0.004)
    print("FFFFFFFFFFFFFFFFFFFFFFFFF",solenoid_ze[1])
    pipe=pipe1+pipe2
    #pipe=pipe1+pipe2+pipe3


    scraper = wp.ParticleScraper(pipe)


    conductors=pipe


    # set up particle termination at cylindrical wall
    wp.top.prwall = solenoid_radius


  
    var1=float(params)  # number of grid cells in x and y direction
    wp.w3d.nx = n_grid
    wp.w3d.ny = n_grid

    solenoid_radius1=solenoid_radius*10.0
    wp.w3d.xmmax =  solenoid_radius1  # x-grid max limit [m]
    wp.w3d.xmmin = -solenoid_radius1  # x-grid min limit [m]
    wp.w3d.ymmax =  solenoid_radius1  # y-grid max limit [m]
    wp.w3d.ymmin = -solenoid_radius1  # y-grid min limit [m]
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
    wp.wxy.ds = 0.1250e-3            # ds for part adv [m]
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

    sort_circular(beam_species)

    for i in range(nsteps):
     
     wp.step()
     
     print("Paso= ",i," ",int(100*(i/nsteps)),"%"," N ",len(electrons.getx()))
     #salvar DATOS PARA GUARDAR CADA PASO
     if False:
            x_aux=electrons.getx()
            xp_aux=electrons.getxp()
            figfd2=plt.figure()
            plt.scatter(x_aux, xp_aux)
            
            for var1 in electrons.getvx():
                print(var1)
            plt.show()
            
                        
            #tamanox=np.std(x_aux)
            e_rms=calc_emit_rms(x_aux, xp_aux)
            print("emitance1===== ",e_rms,np.std(x_aux))
     if i % 5==0 and plot_or_not:


     #if i % 100==0 :


        ppp = wp.getrho(iz=0,solver=wp.getregisteredsolver())
        sumarho=np.sum(ppp)/((n_grid+1)*(n_grid+1))
        x_1 = np.linspace(-solenoid_radius,solenoid_radius,n_grid+1)
        y_1 = np.linspace(-solenoid_radius,solenoid_radius,n_grid+1)
        x4,y4 = np.meshgrid(x_1,y_1)
        r4= (x4**2 + y4**2)
        rmed =np.sum(ppp*r4)/np.sum(ppp)
        xhi= 6*((rmed/(0.00182)))
        rad=electrons.getr()               
        
        #salvar DATOS PARA GUARDAR CADA PASO
        x_aux=electrons.getx()
        xp_aux=electrons.getxp()
        tamanox=np.std(x_aux)
        e_rms=calc_emit_rms(x_aux, xp_aux)
        x_rms.append(tamanox)


        p1x.append(electrons.getx()[10])
        p1y.append(electrons.gety()[10])
        p1vx.append(electrons.getvx()[10])
        p1vy.append(electrons.getvy()[10])
        
        p1xp.append(xp_aux[10])
        p1b.append(electrons.getbz()[10])


        z_posi.append(i*wp.wxy.ds*1e3)
        x_emit.append(1e6*e_rms)
        rhoprome.append(sumarho)
        trans.append( len(x_aux))
        Npart_time.append(len(x_aux))
        chi.append(xhi)
        beta3=tamanox*tamanox/e_rms
        #/e_rms
        phasead.append(beta3)
        #phasead.append(20*wp.wxy.ds/(beta3*2*3.1416))
        #print(Npart_time)
        lineaf2="../salida/profiles_%.4f_.png" % (i)
        if i % 5==0:
            #multiplot2(beam_species,lineaf2)
            multiplot2_zones(beam_species,lineaf2,3,NParticles)
     if i==nsteps-1:
        #sumarho=np.sum(ppp)/((n_grid+1)*(n_grid+1))
        x_1 = np.linspace(-solenoid_radius,solenoid_radius,n_grid+1)
        y_1 = np.linspace(-solenoid_radius,solenoid_radius,n_grid+1)
        x4,y4 = np.meshgrid(x_1,y_1)
        r4= np.sqrt(x4**2 + y4**2)
        #rmed =np.sum(ppp*r4)/np.sum(ppp)
        print(np.sum(r4), np.mean(electrons.getr()))
        
        #print(sumarho)
        #print( "RRRRRRRRRRRRRRRRRRRRRRR", rmed)
        #print(beam_species)
        #print(wp.top.ns,nsteps,i)
        #print(wp.getpid())
        #wp.window(0)
        x_rms2.append(np.std(electrons.getx()))
        z_posi2.append(i*wp.wxy.ds*1e3)
        sizex=(np.std(electrons.getx()))
        e_rms=calc_emit_rms(electrons.getx(),electrons.getxp())
        #z_po=i*wp.wxy.ds*1e3
        #print("campo y tamano",mag_solenoid," ",sizex,len(electrons.getx()),z_po,(np.mean(electrons.getz())))
        #print("Beam species ",beam_species)
        print("salida",sizex,e_rms)
    
    if(plot_or_not):    
       
       #plt.scatter(z_posi.append(i*wp.wxy.ds*1e3),rmed)
       fig, (ax1, ax2,ax3) = plt.subplots(3,1)
       #ax1.scatter(z_posi, x_emit)
       ax1.scatter(z_posi, p1b)


       print(x_emit)
       print(z_posi)
       #for zi, ze in zip(solenoid_zi, solenoid_ze):
       #  plt.axvline(zi, ls="--", color='black')
       #  plt.axvline(ze, ls="--", color='black')


       plt.xlabel("z (m)")
       plt.ylabel("B field")
       plt.title("RMS emit")

       plt.legend()
       #plt.xlim(0.0, run_length)
       #plt.ylim(0.0, 50.08)
        
       ax2.plot(z_posi, x_rms, label="{}, X".format(electrons.name))
       #ax2.plot(z_posi, p1x, label="{}, X".format(electrons.name))


       plt.xlabel("z (mm)")
       plt.ylabel("x (m)")
       plt.title("RMS Envelopes")
       plt.legend()


       #ax3.plot(z_posi, Npart_time, label="{}, X".format(electrons.name))
       ax3.plot(z_posi,Npart_time , label="{}, X".format(electrons.name))
       plt.xlabel("z (mm)")
       plt.ylabel("chi (m)")
       plt.legend()
       
       
       fig10=plt.figure()


       plt.scatter(electrons.getx(), electrons.getxp(), label="{}, X".format(electrons.name))
       plt.xlabel(" x (m)")
       plt.ylabel("xp (rad)")
       plt.legend()
       fig11=plt.figure()


       plt.scatter(z_posi, phasead, label="{}, X".format(electrons.name))
       plt.xlabel(" z (mm)")
       plt.ylabel("phase (a)")
       plt.legend()
       print("phase advance is", np.sum(20*wp.wxy.ds/(beta3*2*3.1416)))
       fig12=plt.figure()


       plt.scatter(p1x,p1xp , label="una part")
       plt.xlabel(" x (mm) 1 part")
       plt.ylabel("xp (a)")
       plt.legend()
       print("phase advance is", np.sum(20*wp.wxy.ds/(beta3*2*3.1416)))
       
       




    
    #input()
    f11 = open('output.txt', 'a')
    f11.write("%.4f\t%.3e\t%.3e" % (var1, sizex,e_rms ))
    f11.write("\n" )
    f11.close()


    
    with open('arreglos.txt', 'w') as file:
        for a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11 in zip(z_posi, x_emit, x_rms, phasead,chi,p1x,p1vx,p1y,p1vy,p1b,trans):
            file.write(f"{a1}\t{a2}\t{a3}\t{a4}\t{a5}\t{a6}\t{a7}\t{a8}\t{a9}\t{a10}\t{a11}\n")
    if(plot_or_not):    
        plt.show()
       


    #return sizex
#except unrecognized arguments:
 #   pass
