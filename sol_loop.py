from curses.panel import top_panel
import matplotlib.pyplot as plt
import numpy as np
from libreria import multiplot, multiplot2, calc_rms, calc_emit_rms, multiplot_magnetized
import scipy.optimize as optimize
#from corrida import f1
import gc
import os,sys, inspect
import tracemalloc
import subprocess


#iport warp as wp

def print_classes():
   for name, obj in inspect.getmembers(sys.modules[__name__]):
       if inspect.isclass(obj):
           print(obj)
print_classes()

gc.collect()
#input()
x_rms=[]
emory=[]
sol1=range(4,30,1)
sol1=np.linspace(0.06, 0.12, 10)
sol1=np.linspace(0.07, 0.071, 10)
preparar="rm output.txt"
os.system(preparar)

preparar="rm *.cgm"
os.system(preparar)

for xsol in sol1:
   #for xsol2 in sol1: 
    tracemalloc.start()
    #sizes=f1(xsol)
    #linea="python3 corrida.py "+str(xsol*0.01)
    #linea="python3 corrida_sem_opt.py "+str(xsol*0.10)+" "+str(xsol2*0.10)
    #linea="python3 corrida_nobeam_aperture.py " + str(0.051) + " " + str(xsol*1.0)
    linea="python3 diseno1.py " + str(0.051) + " " + str(xsol*1.0)


    #este corre el programa
    print(linea)
    subprocess.call(linea, shell=True)
    #os.system(linea)
        
    #memory.append(tracemalloc.get_traced_memory())
    #print("memoria ",tracemalloc.get_traced_memory())
    # stopping the library
    tracemalloc.stop()
    #print("size z",sizes)
    #print("xsol ",xsol)
    #x_rms.append(sizes)
    gc.collect()

A = np.loadtxt('output.txt',skiprows=1)
Z=1*np.array(A[:,0])
XRMS=1*np.array(A[:,1])
fige=plt.figure()
ax=fige.add_subplot(111,title="fff", rasterized=True)
ax.set_xlabel('Z [m]',fontsize=20)
ax.set_ylabel('1 R.M.S [m] ',fontsize=20)
#ax.plot(sol1, emory)
ax.plot(Z,XRMS)

plt.show()
