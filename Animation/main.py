import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(7, 6.5)

ax = plt.axes(xlim=(0, 100), ylim=(0, 100))
patchA = plt.Circle((5, -5), 0.75, fc='y')
patchB = plt.Circle((5, -5), 0.75, fc='y')

G = 6.67 # 6,67408 × 10-11 m3 kg-1 s-2
mA = 2
mB = 2
Dt = 1

vxA = 0.05
vyA = -0.3
vxB = -0.05
vyB = 0.3

vxA = 0.05
vyA = -0.1
vxB = -0.05
vyB = 0.1

def init():
    patchA.center = (20, 50)
    patchB.center = (80, 50)
    
    ax.add_patch(patchA)
    ax.add_patch(patchB)
    
    return patchA,patchB,

def animate(i):
    xA, yA = patchA.center
    xB, yB = patchB.center

    d = np.sqrt((xA-xB)*(xA-xB)+(yA-yB)*(yA-yB))
    
    global vxA
    global vyA
    global vxB
    global vyB
    
    vxA = vxA -G*mB*(xA-xB)*Dt/(d*d*d)
    vyA = vyA -G*mB*(yA-yB)*Dt/(d*d*d)
    
    vxB = vxB -G*mA*(xB-xA)*Dt/(d*d*d)
    vyB = vyB -G*mA*(yB-yA)*Dt/(d*d*d)    
                
    xA = xA + Dt*vxA
    yA = yA + Dt*vyA
    
    patchA.center = (xA, yA)
    
    xB = xB + Dt*vxB
    yB = yB + Dt*vyB

    patchB.center = (xB, yB)
    
    return patchA,patchB,

anim = animation.FuncAnimation(fig, animate,
                               init_func=init,
                               frames=10000,
                               interval=0.1,
                               blit=True,
                               repeat=False)

plt.show()
