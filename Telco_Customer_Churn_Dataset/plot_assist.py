import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def index_as(arr1,arr2):
    return np.where(np.expand_dims(arr1,axis=0) == np.array(arr2).reshape(-1,1))[1]

def bar_plot(df,feature1,feature2,percentages = True,figsize = (14,6),ylabel='',order=None,color='seaborn'):
    plt.figure(figsize = figsize)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    matrix = df.groupby(feature1)[feature2].value_counts().unstack()
    matrix[matrix.isnull()] = 0
    matrix['total'] = matrix.sum(axis=1)
    if isinstance(order,list):
        matrix = matrix.iloc[index_as(matrix.index,order)]
    elif order in matrix.columns:
        matrix.sort_values(order,inplace=True)
    else:
        matrix.sort_index(inplace=True)

    seaborn = np.array(['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'])
    custom = np.array(['#ff0000','#00ff00','#00ffff','#0000ff','#ffff00','#ff00ff','#000000'])
    colors = {'custom':custom,'seaborn':seaborn}[color]

    idxs = range(matrix.index.shape[0])
    #plt.bar(x = idxs,height = matrix.iloc[:,0],width = 0.5, label=matrix.iloc[:,0].name,color =colors[i] + '70',edgecolor=colors[i])
    for i in range(0,matrix.columns.shape[0] - 1):
        plt.bar(x = idxs,height = matrix.iloc[:,i],width = 0.5,bottom=np.sum(matrix.iloc[:,:i].values,axis=1) if i else None,label=matrix.iloc[:,i].name,color =colors[i] + '70',edgecolor=colors[i])
    if percentages:
        ypos = np.sum(plt.ylim())
        for i in idxs:
            for j,val in enumerate(np.round(matrix.iloc[i,:-1]/matrix['total'].iloc[i]*100,2)):
                plt.text(x = idxs[i] - 0.3,y = ypos*(j*15 + 10)/100,s=f'{val}%',rotation=90, horizontalalignment='center',verticalalignment='center',color=colors[j])
            plt.plot([idxs[i],idxs[i]],[0,matrix['total'].iloc[i]],color ='#2ca02c',ls='--',lw=2)
            plt.text(x = idxs[i] - 0.3,y = ypos*((j+1)*15 + 10)/100,s=f"{np.round(matrix['total'].iloc[i]/matrix['total'].sum()*100,2)}%",rotation=90, horizontalalignment='center',verticalalignment='center',color='green')
    plt.legend(title=matrix.columns.name)
    plt.xlabel(matrix.index.name)
    plt.xlim(idxs[0] - 0.5,idxs[-1] + 0.5)
    plt.title(f'{matrix.index.name} vs. {matrix.columns.name}')
    plt.ylabel(ylabel)
    plt.gca().set_axisbelow(True)
    plt.grid(axis='y', alpha=0.75)
    plt.xticks(ticks=idxs,labels=matrix.index,rotation=90)
    plt.show()

def log10(X,inv=False):
    if inv:
        return 10 ** X
    return np.log10(X)

def logit(X,inv=False):
    if inv:
        return  np.exp(X) / (1. + np.exp(X))
    return np.log(X) - np.log(1. - X)

def linear(X,inv=False):
    return X

def sqrt(X,inv=False):
    if inv:
        return X ** 2
    return np.sqrt(X)

def plt_hist(column1,column2,figsize=(14,6),xlim=None,bins=None,scale='linear',color='seaborn',histtype='stepfilled'):
    transform = {'log':log10,'logit':logit,'sqrt':sqrt,'linear':linear}[scale]
    scaled_col = transform(column2)
    plt.figure(figsize=figsize)

    seaborn = np.array(['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'])
    custom = np.array(['#ff0000','#00ff00','#00ffff','#0000ff','#ffff00','#ff00ff','#000000'])
    colors = {'custom':custom,'seaborn':seaborn}[color]

    for i,color in zip(column1.unique(),colors):
        plt.hist(x=scaled_col[column1 == i],color=color + '70',label=i,bins=bins,histtype=histtype,ec=color)
        col_mean = scaled_col[column1 == i].mean()
        plt.axvline(x=col_mean,color=color,ls='--',label=f'{i} mean: {np.round(transform(col_mean,inv=True),2)}')
    
    plt.legend(title=column1.name)
    plt.xlabel(scaled_col.name)
    plt.ylabel('frequency')
    plt.title(f'{column1.name} vs. {scaled_col.name} ({scale} scale)')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    if not xlim:
        xlim = (column2.min(),column2.max())
    plt.xlim(transform(xlim))
    ticks = np.linspace(*transform(xlim),11)
    plt.xticks(ticks,labels=np.round(transform(ticks,inv=True),2),rotation=45)
    plt.show()

def pie_plot(column,color='seaborn'):
    seaborn = np.array(['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'])
    custom = np.array(['#ff0000','#00ff00','#00ffff','#0000ff','#ffff00','#ff00ff','#000000'])
    colors = {'custom':custom,'seaborn':seaborn}[color]
    [c + '70' for c in colors]

    total = column.value_counts().sort_index()
    perc = np.round(total/np.sum(total)*100,2)
    plt.figure(figsize = (15,5))
    explode= np.arange(0,total.shape[0]*0.01,step=0.01)
    labels = [f'{i} - {j:1.2f} %' for i,j in zip(perc.index,perc.values)]
    plt.pie(perc,startangle = 90,shadow=True,colors= colors,explode=explode)
    plt.legend(title = 'info.',labels =labels,loc='center left', bbox_to_anchor=(1, 0.5))
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


def simulate_bootstrap_fast(n,p):
    samples1 = np.random.binomial(n[0],p[0], 10000)/n[0]
    samples2 = np.random.binomial(n[1],p[1], 10000)/n[1]
    return samples2 - samples1

def simulate_bootstrap_slow(n,p):
    diffs = []
    for _ in range(10000):
        sample1 = np.random.choice([1,0], size = n[0], p = [p[0], (1- p[0])]).mean()
        sample2 = np.random.choice([1,0], size = n[1], p = [p[1], (1- p[1])]).mean()
        diffs.append(sample2 - sample1)
    return np.array(diffs)