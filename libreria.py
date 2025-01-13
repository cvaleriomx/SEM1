import matplotlib.pyplot as plt
import numpy as np

__author__ = "Daniel Winklehner v1 cristhian valerio v2 "
__doc__ = """
Warp Script with a simple solenoid lattice to demonstrate space charge compensation.

USPAS Jan 2018
"""
def calc_rms(protons):


        x1=protons.getx()
        MX=np.mean(x1)
        X=x1
        #print("media = ",MX)
        varx=0
        LS=len(X)
        for i2 in range(0,len(X)):
            varx   = varx   + (X[i2]-MX) *(X[i2]-MX)/LS
        RMS=np.sqrt(varx*varx)
        return(RMS)

def calc_emit_rms(x1,xp1):
    MX=np.mean(x1)
    MPX=np.mean(xp1)
    X=x1
    PX2=xp1
    print("media = ",MX)
    rmd=np.sqrt(np.mean(x1**2))
    print("media = ",1e3*rmd)
    LS=len(xp1)
    varx=0
    varpx=0
    varxpx=0
    for i2 in range(0,len(X)):
              	varx   = varx   + (X[i2]-MX) *(X[i2]-MX)/LS
              	varpx  = varpx  + (PX2[i2]-MPX)*(PX2[i2]-MPX)/LS
              	varxpx = varxpx + (X[i2]-MX)*(PX2[i2]-MPX)/LS
    print(varx)
    e_rms = 1*np.sqrt(varx*varpx-varxpx*varxpx)
    print ("RMS Size X = %.4f mm Emittance =  %03s mm.mrad" % (np.sqrt(varx)*1e3,e_rms*1000000))
    return (e_rms)


def multiplot(beam_species):

    if not isinstance(beam_species, list):
        beam_species = [beam_species]
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax4 = plt.subplot2grid((2, 2), (0, 1))
    #ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax3 = plt.subplot2grid((2, 2), (1, 1))

    for species in beam_species:

        ax2.scatter(1e3*species.getx(), 1e3*species.gety(),
                    s=0.5, label="{}".format(species.name))
        #ax3.scatter(1e3*species.getx(), 1e3*species.getxp(),
         #           s=0.5, label="{}".format(species.name))

    plt.sca(ax2)
    plt.legend
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Cross-Section")
    
    plt.sca(ax3)
    Hx, xxedges, yxedges = np.histogram2d(1e3*species.getx(),1e3*species.getxp(),bins=100)
    Hx = np.rot90(Hx)
    Hx = np.flipud(Hx)
    Hmaskedx = np.ma.masked_where(Hx==0,Hx)
    plt.pcolormesh(xxedges,yxedges,Hmaskedx)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.legend
    plt.xlabel("x (mm)")
    plt.ylabel("x' (mrad)")
    plt.title("XX' Phase Space")
    
    return plt.gcf(), [ax1, ax2, ax3,ax4]
def multiplot2(beam_species,lineaf2):

    if not isinstance(beam_species, list):
        beam_species = [beam_species]
   
    ax12 = plt.subplot(2, 1, 1)
    ax13 = plt.subplot(2, 1, 2)
    for species in beam_species:

        ax12.scatter(1e3*species.getx(), 1e3*species.gety(),
                    s=0.5, label="{}".format(species.name))
        #ax13.scatter(1e3*species.getx(), 1e3*species.getxp(),
        #            s=0.5, label="{}".format(species.name))

    
    plt.sca(ax12)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.legend
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Cross-Section")
    
    plt.sca(ax13)
    #ax13=fig2.add_subplot(111,title="Y vs Y\'", rasterized=True)
    limits=[[-50, 50], [-70, 70]]
    #Hx, xxedges, yxedges = np.histogram2d(1e3*species.getx(),1e3*species.getxp(),bins=100,range=limits)
    Hx, xxedges, yxedges = np.histogram2d(1e3*species.getx(),1e3*species.getxp(),bins=100)

    Hx = np.rot90(Hx)
    Hx = np.flipud(Hx)
    Hmaskedx = np.ma.masked_where(Hx==0,Hx)
    plt.pcolormesh(xxedges,yxedges,Hmaskedx)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    #ax.set_xlabel('Y [mm]',fontsize=20)
    #ax.set_ylabel('Y\' [mrad] ',fontsize=20)
    #ax.grid(True)
    #plt.xlim(-50, 50)
    #plt.ylim(-100, 100)
    plt.clim(0, 14) 
    plt.legend
    plt.xlabel("x (mm)")
    plt.ylabel("x' (mrad)")
    plt.title("XX' Phase Space")
    plt.savefig(lineaf2)
    plt.clf()

def multiplot2_zones(beam_species,lineaf2,zonas):

    if not isinstance(beam_species, list):
        beam_species = [beam_species]
   # Dividir las partículas en zonas según su índice
    Npa=np.size(beam_species[0].getx())
    zonas_indices = np.array_split(np.arange(Npa), zonas)

        # Colores para las zonas
    colores = plt.cm.viridis(np.linspace(0, 1, zonas))


    ax12 = plt.subplot(2, 1, 1)
    ax13 = plt.subplot(2, 1, 2)
    for species in beam_species:
        x1=1e3*species.getx()
        y1=1e3*species.gety()

       # ax12.scatter(1e3*, 1e3*,
        #            s=0.5, label="{}".format(species.name))
        
        
        for i, indices in enumerate(zonas_indices):
            ax12.scatter(
                x1[indices],
                y1[indices],
                color=colores[i],
                label=f'Zona {i + 1}',
                s=1,alpha=0.6
            )

        #ax13.scatter(1e3*species.getx(), 1e3*species.getxp(),
        #            s=0.5, label="{}".format(species.name))

    
    plt.sca(ax12)
    #plt.xlim(-50, 50)
    #plt.ylim(-50, 50)
    plt.legend
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Cross-Section")
    
    plt.sca(ax13)
    #ax13=fig2.add_subplot(111,title="Y vs Y\'", rasterized=True)
    limits=[[-50, 50], [-70, 70]]
    Hx, xxedges, yxedges = np.histogram2d(1e3*species.getx(),1e3*species.getxp(),bins=100,range=limits)
    Hx = np.rot90(Hx)
    Hx = np.flipud(Hx)
    Hmaskedx = np.ma.masked_where(Hx==0,Hx)
    plt.pcolormesh(xxedges,yxedges,Hmaskedx)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    #ax.set_xlabel('Y [mm]',fontsize=20)
    #ax.set_ylabel('Y\' [mrad] ',fontsize=20)
    #ax.grid(True)
    #plt.xlim(-50, 50)
    #plt.ylim(-100, 100)
    plt.clim(0, 14) 
    plt.legend
    plt.xlabel("x (mm)")
    plt.ylabel("x' (mrad)")
    plt.title("XX' Phase Space")
    plt.savefig(lineaf2)
    plt.clf()

def multiplot_magnetized(protons,lineaf2):
    xp1=protons.getxp()
    x1=protons.getx()
    y1=protons.gety()
    
    px1=protons.getvx()*1.67e-27
    MX=np.mean(x1)
    bz=wp.getbz()
    px2=px1+1.6e-19*bz*y1/2
    print("Media ",np.mean(px1),np.mean(px2))
    print("STD ",np.std(px1),np.std(px2))
    ax12 = plt.subplot(2, 1, 1)
    ax13 = plt.subplot(2, 1, 2)
    ax12.scatter(1e3*x1,px1,s=0.5)
        #ax13.scatter(1e3*species.getx(), 1e3*species.getxp(),
        #            s=0.5, label="{}".format(species.name))

    
    plt.sca(ax12)
    plt.xlim(-50, 50)
    #plt.ylim(-50, 50)
    plt.legend
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Cross-Section")

    ax13.scatter(1e3*x1,px2,s=0.5)
    plt.sca(ax13)
    plt.xlim(-50, 50)
    #plt.ylim(-50, 50)
    plt.legend
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Cross-Section")
    
    
    plt.savefig(lineaf2)
    plt.clf()
    
    #return plt.gcf(), [ax2, ax3]

# 1. Definir la función parabólica (PDF no normalizada)
def pdf_parabola(x, a, c):
    #return a * x*2 + c
    return 1+ a *(c* x**2 -1)

# 2. Normalizar la función
def normalizar_pdf_parabola(a, c, rango):
    x = np.linspace(rango[0], rango[1], 1000)
    y = pdf_parabola(x, a, c)
    area = np.trapz(y, x)
    return lambda x: pdf_parabola(x, a, c) / area

# 3. Método de Rechazo para generar muestras
def generar_muestras_parabola(pdf, rango, num_muestras):
    muestras = []
    while len(muestras) < num_muestras:
        x = np.random.uniform(rango[0], rango[1])
        y = np.random.uniform(0, max(pdf(np.array([rango[0], rango[1]]))))
        if y <= pdf(x):
            muestras.append(x)
    return np.array(muestras)

def generar_circular(arreglo):
    i=0
    muestrasr=[]
    contr=len(arreglo)
    while i<contr:
        theta = np.random.uniform(0, 2*np.pi)
        
        if np.random.uniform(0, 1) <= 3:
            x = arreglo[i] * np.cos(theta)
            y = arreglo[i] * np.sin(theta)
            muestrasr.append((x, y))
            i=i+1
    return np.array(muestrasr)