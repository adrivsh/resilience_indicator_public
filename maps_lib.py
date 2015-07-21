import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from subprocess import call
import numpy as np

def n_to_one_normalizer(s,n=0):
    y =(s-s.min())/(s.max()-s.min())
    return n+(1-n)*y
    
def bins_normalizer(x,n=7):
    n=n-1
    y= n_to_one_normalizer(x,0)  #0 to 1 numbe
    return np.floor(n*y)/n

def quantile_normalizer(column, nb_quantile=5):
    return (pd.qcut(column, nb_quantile,labels=False))/(nb_quantile-1)

def num_to_hex(x):
    h = hex(int(255*x)).split('x')[1]
    if len(h)==1:
        h="0"+h
    return h

def data_to_rgb(serie,color_maper=plt.cm.get_cmap("Blues_r"),normalizer=n_to_one_normalizer,norm_param=0):    
    data_n = normalizer(serie,norm_param).dropna()
    colors = pd.DataFrame(color_maper(data_n),index=serie.index, columns=["r","g","b","a"]).applymap(num_to_hex)
    return "#"+colors.r+colors.g+colors.b
    
    
def append_styles_to_map(outmap,styles,inmap='BlankWorldMap.svg'):
    inFile = open(inmap)
    outFile = open(outmap+".svg", "w")
    buffer = ['']
    isreplacing=False
    for line in inFile:
        if line.startswith('/* Begin country color*/'):
            isreplacing=True
            buffer.append(line)
            buffer.append(styles)
        elif line.startswith('/* End country color*/'):
            isreplacing=False
        if not isreplacing:
            buffer.append(line)
    outFile.write("".join(buffer))
    inFile.close()
    outFile.close()
    
    call("inkscape -f {map}.svg -e {map}.png -d 150".format(map=outmap))

    
def make_legend(serie,label,path):
    fig = plt.figure(figsize=(8,3))
    ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])

    cmap = mpl.cm.get_cmap("Greens")
    vmin=(100*serie.min())
    vmax=(100*serie.max())

    # define the bins and normalize
    bounds =np.linspace(vmin,vmax,7)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


    cb = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='horizontal', format="%2.1f")
    #cb.ax.set_xticklabels(['0.01%','0.1%','1%','10%'])
    cb.set_label(label,size=20)

    plt.savefig(path,bbox_inches="tight",transparent=True)    