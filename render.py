import subprocess
import numpy as np
import shutil

mode = 'growing'
nsec = 5.0
npts = int(nsec * 60)

if mode == 'moving':
    x_pos = np.linspace(0.0, 1.0, npts)
    cmd   = './lensing --x {}'
elif mode == 'growing':
    x_pos = np.logspace(-3, 3, npts, base=10)
    cmd   = './lensing --M {}'

    
for i, x in enumerate(x_pos):
    c = cmd.format(x)
    print(i, c)
    subprocess.call(c.split())
    shutil.move('result.png', 'render/img_{:05}.png'.format(i))
