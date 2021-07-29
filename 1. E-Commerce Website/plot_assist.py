import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def bar_plot(df,feature1,feature2,percentages = True,figsize = (10,6),ylabel=''):
    plt.figure(figsize = figsize)
    matrix = df.groupby(feature1)[feature2].value_counts().unstack()
    matrix[matrix.isnull()] = 0
    total = matrix.sum(axis=1)
    idxs = range(matrix.index.shape[0])
    colors = np.array(['#ff0000','#00ff00','#00ffff','#0000ff','#ffff00','#ff00ff','#000000'])
    plt.bar(x = idxs,height = matrix.iloc[:,0],width = 0.5, label=matrix.iloc[:,0].name,color =colors[0],edgecolor='black')
    for i in range(1,matrix.columns.shape[0]):
        plt.bar(x = idxs,height = matrix.iloc[:,i],width = 0.5,bottom=matrix.iloc[:,i-1] ,label=matrix.iloc[:,i].name,color =colors[i],edgecolor='black')
    if percentages:
        ypos = np.sum(plt.ylim())
        for i in idxs:
            for j,val in enumerate(np.round(matrix.iloc[i]/total.iloc[i]*100,2)):
                plt.text(x = idxs[i] - 0.3,y = ypos*(j*15 + 10)/100,s=f'{val}%',rotation=90, horizontalalignment='center',verticalalignment='center',color=colors[j])
            plt.plot([idxs[i],idxs[i]],[0,total.iloc[i]],color ='green',ls='--',lw=2)
            plt.text(x = idxs[i] - 0.3,y = ypos*((j+1)*15 + 10)/100,s=f'{np.round(total.iloc[i]/total.sum()*100,2)}%',rotation=90, horizontalalignment='center',verticalalignment='center',color='green')
    plt.legend(title=matrix.columns.name)
    plt.xlabel(matrix.index.name)
    plt.xlim(idxs[0] - 0.5,idxs[-1] + 0.5)
    plt.title(f'{matrix.index.name} vs. {matrix.columns.name}')
    plt.ylabel(ylabel)
    plt.gca().set_axisbelow(True)
    plt.grid(axis='y', alpha=0.75)
    plt.xticks(ticks=idxs,labels=matrix.index,rotation=90)
    plt.show()

def pie_plot(column):
    colors = np.array(['#00ff00','#ff0000','#00ffff','#0000ff','#ffff00','#ff00ff','#ff8000'])
    total = column.value_counts().sort_index()
    perc = np.round(total/np.sum(total)*100,2)
    plt.figure(figsize = (15,5))
    explode= np.arange(0,total.shape[0]*0.01,step=0.01)
    labels = [f'{i} - {j:1.2f} %' for i,j in zip(perc.index,perc.values)]
    plt.pie(perc,startangle = 90,shadow=True,colors=colors,explode=explode)
    plt.legend(title = 'info.',labels =labels,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.gca().set_title(total.name)
    plt.show()

def pie_plot2(column):
    colors = np.array(['#00ff00','#ff0000','#00ffff','#0000ff','#ffff00','#ff00ff','#ff8000'])
    total = column.value_counts().sort_index()
    perc = np.round(total/np.sum(total)*100,2)
    plt.figure(figsize = (15,5))
    explode= np.arange(0,total.shape[0]*0.01,step=0.01)
    patches, labels, pct_texts = plt.pie(perc,startangle = 90,rotatelabels=True,pctdistance=1.2,autopct ='%.2f%%',shadow=True,colors=colors,explode=explode,wedgeprops = {'edgecolor':'0'})
    for label, pct_text in zip(labels, pct_texts):
        pct_text.set_rotation(label.get_rotation())
    plt.legend(title = 'info.',labels =total.index,loc='center left', bbox_to_anchor=(1.1, 0.5))
    plt.gca().set_title(total.name)
    plt.show()

def descrip_column(column):
    print('name:',column.name)
    print('dtype:',column.dtype)
    print('null count:',column.isnull().sum())
    print('unique:',column.unique())
    print('unique count:',column.nunique())
    if column.dtype != object:
        print('max range:',column.max())
        print('min range:',column.min())
    vc = column.value_counts()
    print('max frequncy:',vc.max())
    print('min frequncy:',vc.min())


def Proportion(df,feature1,feature2):
    matrix = df.groupby(feature1)[feature2].value_counts().unstack()
    matrix[matrix.isnull()] = 0
    total = np.sum(matrix.values,axis=1)
    data = np.round(matrix/total.reshape(-1,1)*100,2)
    data['total'] = np.round(total/np.sum(total)*100,2)
    return data

def plot_Hypothesis(nullvals,p_diffs,critical_value,figsize = (14,6)):
    plt.figure(figsize = figsize)
    sns.kdeplot(nullvals,color='#ff0000',shade=False,label='Null Hypothesis')
    sns.kdeplot(p_diffs,color='#0000ff',shade=False,label='Alternative Hypothesis')
    xx,yy = plt.gca().lines[0].get_data()
    idx = xx >= critical_value
    xcoords = xx[idx].copy()
    ycoords = yy[idx].copy()
    xcoords = np.append(xcoords, xcoords[0])
    ycoords = np.append(ycoords, 0 )
    plt.fill(xcoords, ycoords,color='#ff000070')
    ########################################
    xx,yy = plt.gca().lines[1].get_data()
    idx = xx <= critical_value
    xcoords = xx[idx].copy()
    ycoords = yy[idx].copy()
    xcoords = np.append(xcoords, xcoords[-1])
    ycoords = np.append(ycoords, 0 )
    plt.fill(xcoords, ycoords,color='#0000ff70')
    #############################################3
    labels = [f'Type I Error (α): {(nullvals > critical_value).mean()}',f'Type II Error (β): {(p_diffs < critical_value).mean()}']
    colors = np.array(['#ff000070','#0000ff70'])
    handles=[plt.Rectangle((0,0),1,1, facecolor=color,edgecolor='#000000',label=f'{label}') for label,color in zip(labels,colors)]
    plt.axvline(critical_value,c='r',ls='--',label = 'Critical Value')
    plt.legend(title='Info.',handles=plt.gca().get_legend_handles_labels()[0] + handles)
    plt.title('Hypothesis Test Sampling Distribution')
    plt.xlabel('Samples')
    plt.show()


def plot_dist(values,color='#ff0000',bins=20,figsize = (14,6)):
    plt.figure(figsize = figsize)
    plt.hist(x=values,color='#70707070',density=True,bins=bins);
    sns.kdeplot(x=values,color=color);
    plt.axvline(values.mean(),c='#707070',ls='--',label ='mean value');
    plt.title('Sampling Distribution')
    plt.xlabel('X')
    plt.legend(title='Info.')
    plt.show()