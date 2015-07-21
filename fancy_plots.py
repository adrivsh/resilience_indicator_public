import matplotlib.pyplot as plt
import pandas as pd

def fancy_barh(data,formater = lambda x:"{:2.1f}".format(100*x)):
    plt.figure(figsize=(9,25))

    pos = range(len(data.index))    
    
    #ensures Series are seen as dataFrames
    if  len(data.shape)==1:
        data = pd.DataFrame(data)
    
    nb_cols =  data.shape[1]
    
    has_legend  = nb_cols>1
    
    #orders datafram acording to first columan
    data=data.sort_index(by=data.columns[0])


    #bars
    for col in data.columns:
        plt.barh(pos,data[col],align="center",color="#a1d99b",edgecolor="#31a354",alpha=1/nb_cols)

    #frame
    plt.ylim(ymin=-0.5,ymax = max(pos)+4 if has_legend else  max(pos)+1); #room for legend
    plt.xlim(xmin=0)
    
    ax=plt.gca()
    #remove spines
    ax.spines['bottom'].set_color("none")
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    #removes xticks
    for tic in ax.yaxis.get_major_ticks()+ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        
    #Yticks
    plt.yticks(pos,data.index);    
        
    #removes xlables
    plt.setp(ax.get_xticklabels(), visible=False); 

    
    #anotations
    is_even   = nb_cols>1
    for col in data.columns:
        
        #All labels
        for i in pos:
            x=data.ix[i,col]
            if x>1/100 or col==data.columns[-1]: #marks big numbers and the last one
                ax.annotate(formater(x),  xy=(x,i),xycoords='data',ha="right" if is_even else "left"
                            ,va="center", size=12,  xytext=(-5 if is_even else 5, -1), textcoords='offset points')
        
        if nb_cols>1 :
            #the "legend"
            ax.annotate(col,  xy=(x,i),xycoords='data',va="center",
              xytext=(-50 if is_even else 50, 35), textcoords='offset points', 
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=-0.3" if is_even else "arc3,rad=0.3",
                                )
                )
            
        is_even = not is_even


