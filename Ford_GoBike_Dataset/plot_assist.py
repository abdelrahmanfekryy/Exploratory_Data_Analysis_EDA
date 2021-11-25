import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def index_as(arr1,arr2):
    return np.where(np.expand_dims(arr1,axis=0) == np.array(arr2).reshape(-1,1))[1]

def bar_plot(df,feature1,feature2,percentages = True,figsize = (10,6),ylabel='',order=None,color='seaborn'):
    plt.figure(figsize = figsize)
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
    plt.bar(x = idxs,height = matrix.iloc[:,0],width = 0.5, label=matrix.iloc[:,0].name,color =colors[0],edgecolor='black')
    for i in range(1,matrix.columns.shape[0] - 1):
        plt.bar(x = idxs,height = matrix.iloc[:,i],width = 0.5,bottom=np.sum(matrix.iloc[:,:i].values,axis=1) ,label=matrix.iloc[:,i].name,color =colors[i],edgecolor='black')
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

##################################################################################################
def barh_plot(df,feature1,feature2,percentages = True,figsize = (6,10),xlabel='',order=None,color='seaborn'):
    plt.figure(figsize = figsize)
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

    plt.barh(y = idxs,width= matrix.iloc[:,0],height= 0.5, label=matrix.iloc[:,0].name,color =colors[0],edgecolor='black')
    for i in range(1,matrix.columns.shape[0] - 1):
        plt.barh(y = idxs,width= matrix.iloc[:,i],height= 0.5,left=np.sum(matrix.iloc[:,:i].values,axis=1) ,label=matrix.iloc[:,i].name,color =colors[i],edgecolor='black')
    if percentages:
        xpos = np.sum(plt.xlim())
        for i in idxs:
            for j,val in enumerate(np.round(matrix.iloc[i,:-1]/matrix['total'].iloc[i]*100,2)):
                plt.text(y = idxs[i] + 0.3,x = xpos*(j*15 + 10)/100,s=f'{val}%', horizontalalignment='center',verticalalignment='center',color=colors[j])
            plt.plot([0,matrix['total'].iloc[i]],[idxs[i],idxs[i]],color ='green',ls='--',lw=2)
            plt.text(y = idxs[i] + 0.3,x = xpos*((j+1)*15 + 10)/100,s=f'{np.round(matrix["total"].iloc[i]/matrix["total"].sum()*100,2)}%', horizontalalignment='center',verticalalignment='center',color='green')
    plt.legend(title=matrix.columns.name)
    plt.ylabel(matrix.index.name)
    plt.ylim(idxs[0] - 0.5,idxs[-1] + 0.5)
    plt.title(f'{matrix.index.name} vs. {matrix.columns.name}')
    plt.xlabel(xlabel)
    plt.gca().set_axisbelow(True)
    plt.grid(axis='x', alpha=0.75)
    plt.yticks(ticks=idxs,labels=matrix.index)
    plt.show()

###################################################################################################################################

def pie_plot(column,spacing = 0.01,shadow=True,figsize = (15,5),color='seaborn'):
    seaborn = np.array(['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'])
    custom = np.array(['#ff0000','#00ff00','#00ffff','#0000ff','#ffff00','#ff00ff','#000000'])
    colors = {'custom':custom,'seaborn':seaborn}[color]

    total = column.value_counts().sort_index()
    perc = np.round(total/np.sum(total)*100,2)
    plt.figure(figsize = figsize)
    explode= np.arange(0,total.shape[0])*spacing
    labels = [f'{i} - {j:1.2f} %' for i,j in zip(perc.index,perc.values)]
    plt.pie(perc,startangle = 90,colors=colors,explode=explode,shadow=shadow)
    plt.legend(title = 'info.',labels =labels,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.gca().set_title(total.name)
    plt.show()

#############################################################################################

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


def union(arr1,arr2,numpy=False):
    if numpy:
        return np.union1d(arr1,arr2)
    else:
        return np.array(list(set(arr1) | set(arr2)))


def intersection(arr1,arr2,numpy=False):
    if numpy:
        return np.intersect1d(arr1,arr2)
    else:
        return np.array(list(set(arr1) & set(arr2)))

def symmetric_difference(arr1,arr2,numpy=False):
    if numpy:
        return np.setdiff1d(arr1,arr2)
    else:
        return np.array(list(set(arr1) ^ set(arr2)))


###########################################################################################################3
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
    #plt.xticks(plt.xticks()[0],labels=np.round(transform(plt.xticks()[0],inv=True),2),rotation=45)
    plt.xticks(ticks,labels=np.round(transform(ticks,inv=True),2),rotation=45)
    plt.show()
######################################################################################################
def box_plot(df,feature1,feature2,figsize = (14,6),ylim=None,showfliers=True,order=None):
    plt.figure(figsize=figsize)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    data = df.groupby(feature1)[feature2].apply(np.array)

    if isinstance(order,list):
        data = data.iloc[index_as(data.index,order)]
    else:
        data.sort_index(inplace=True)

    medianprops = {'color':'#000000','ls':'-.'}
    meanprops = {'color':'#000000','ls':'--'}
    boxprops = {'facecolor':'#1f77b470','edgecolor':'#1f77b4'}
    whiskerprops = {'ls':'--'}
    
    for i,val in enumerate(data.index):
        plt.boxplot(x=data[val],vert=True,widths=0.8,manage_ticks=True,meanline=True,showmeans=True,patch_artist=True,positions=[i],
        medianprops=medianprops,meanprops=meanprops,boxprops=boxprops,whiskerprops=whiskerprops,showfliers=showfliers);
    plt.gca().set_xticklabels(data.index)
    plt.title(f'{data.name} vs {data.index.name}')
    plt.ylabel(data.name)
    plt.xlabel(data.index.name)
    plt.ylim(ylim)
    plt.show()
#########################################################################################################
def box_plot2(df,feature1,feature2,figsize = (10,6),ylim=None):
    plt.figure(figsize=figsize)
    medianprops = {'color':'#00000000'}
    meanprops = {'color':'#ff0000','ls':'-'}
    whiskerprops = {'ls':'--'}
    data = df.groupby(feature1)[feature2].apply(np.array)
    colors = np.array(['#ff0000','#00ff00','#00ffff','#0000ff','#ffff00','#ff00ff','#000000'])
    for i,val in enumerate(data.index):
        boxprops = {'facecolor':'#00000000','edgecolor':colors[i],'label':val}
        plt.boxplot(x=data[val],vert=True,widths=0.8,manage_ticks=True,meanline=True,showmeans=True,patch_artist=True,positions=[i],medianprops=medianprops,meanprops=meanprops,whiskerprops=whiskerprops,boxprops=boxprops);
    plt.gca().set_xticklabels(data.index)
    plt.title(f'{data.name} vs {data.index.name}')
    plt.ylabel(data.name)
    plt.xlabel(data.index.name)
    plt.legend(title='info.')
    plt.ylim(ylim)
    plt.show()
#####################################################################################################
def plt_hist_uni(column,figsize=(14,6),xlim=None,bins=None,scale='linear',color='seaborn'):
    transform = {'log':log10,'logit':logit,'sqrt':sqrt,'linear':linear}[scale]
    scaled_col = transform(column)
    plt.figure(figsize=figsize)

    seaborn = np.array(['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'])
    custom = np.array(['#ff0000','#00ff00','#00ffff','#0000ff','#ffff00','#ff00ff','#000000'])
    colors = {'custom':custom,'seaborn':seaborn}[color]

    plt.hist(x=scaled_col,color=colors[0] + '70',bins=bins,histtype='stepfilled',ec=colors[0])
    col_mean = scaled_col.mean()
    plt.axvline(x=col_mean,color=colors[0],ls='--',label=f'mean: {np.round(transform(col_mean,inv=True),2)}')
    
    plt.legend(title='Info.')
    plt.xlabel(scaled_col.name)
    plt.ylabel('frequency')
    plt.title(f'{scaled_col.name} ({scale} scale)')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    if not xlim:
        xlim = (column.min(),column.max())
    plt.xlim(transform(xlim))
    ticks = np.linspace(*transform(xlim),11)
    plt.xticks(ticks,labels=np.round(transform(ticks,inv=True),2),rotation=45)
    plt.show()

#########################################################################################

def bar_plot_uni(column,vert = False,percentages = True,figsize = (10,6),axlabel='',order=None,color='#1f77b4'):
    plt.figure(figsize = figsize)
    data = column.value_counts()
    data[data.isnull()] = 0

    if isinstance(order,list):
        data = data.iloc[index_as(data.index,order)]
    elif order == 'total':
        data.sort_values(inplace=True)
    else:
        data.sort_index(inplace=True)

    data2 = np.round(data/data.sum()*100,2)
    idxs = range(data.index.shape[0])

    if vert:
        plt.barh(y = idxs,width = data,height = 0.5,color = color + '70',edgecolor=color)
        plt.grid(axis='x', alpha=0.75)
        plt.yticks(ticks=idxs,labels=data.index)
        plt.ylabel(data.name)
        plt.xlabel(axlabel)
        plt.ylim(idxs[0] - 0.5,idxs[-1] + 0.5)
        if percentages:
            for i in idxs:
                plt.text(y = i + 0.3,x = np.sum(plt.xlim())*20/100,s=f'{data2.iloc[i]}%', horizontalalignment='center',verticalalignment='center',color='k')
                           
    else:
        plt.bar(x = idxs,height = data,width = 0.5,color = color + '70',edgecolor=color)
        plt.grid(axis='y', alpha=0.75)
        plt.xticks(ticks=idxs,labels=data.index,rotation=90)
        plt.xlabel(data.name)
        plt.ylabel(axlabel)
        plt.xlim(idxs[0] - 0.5,idxs[-1] + 0.5)
        if percentages:
            for i in idxs:
                plt.text(x = i - 0.3,y = np.sum(plt.ylim())*20/100,s=f'{data2.iloc[i]}%',rotation=90, horizontalalignment='center',verticalalignment='center',color='k')
                
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.title(data.name)
    plt.gca().set_axisbelow(True)
    plt.show()

###############################################################################33

def box_plot_multi(df,feature1,feature2,hue,figsize = (10,6),ylim=None,showfliers=True,order=None):
    plt.figure(figsize=(14,6))
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    data = df.groupby([feature1,hue])[feature2].apply(np.array).unstack()

    if isinstance(order,list):
        data = data.iloc[index_as(data.index,order)]
    else:
        data.sort_index(inplace=True)

    colors = np.array(['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'])
    medianprops = {'color':'#000000','ls':'-.'}
    meanprops = {'color':'#000000','ls':'--'}
    for j in range(data.columns.shape[0]):
        boxprops = {'facecolor':colors[j] + '70','edgecolor':colors[j]}
        plt.boxplot(x=data.iloc[:,j],vert=True,widths=0.8,manage_ticks=True,meanline=True,showmeans=True,patch_artist=True,positions=np.arange(data.index.shape[0])*data.columns.shape[0] + j
        ,boxprops =boxprops,medianprops=medianprops,meanprops=meanprops,showfliers=showfliers)
        
    plt.xticks(np.arange(data.index.shape[0])*data.columns.shape[0] + (data.columns.shape[0] - 1)/2,labels=data.index,rotation=90)
    plt.title(f'{feature1} vs. {feature2}')
    plt.xlabel(data.index.name)
    plt.ylabel(feature2)
    handles=[plt.Rectangle((0,0),1,1, facecolor=colors[i] + '70',edgecolor=colors[i],label=label) for i,label in enumerate(data.columns)]
    plt.legend(title=data.columns.name,handles=handles)
    plt.ylim(ylim)
    plt.show()

