import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import  StandardScaler
import pickle
import numpy as np
from sklearn.decomposition import PCA

UNCLASSIFIED=0
NOISE=-1
#计算每一个数据点与其他每个点之前的距离
def getdatadisstance(datas):
    line,column=np.shape(datas)#返回datas里面列,行的数目
    dists=np.zeros([line,line])#创建一个初始值为0 column*column的矩阵
    for i in range(0,line):
        for j in range(0, line):
            vi=datas[i,:]
            vj=datas[j,:]
            #通过把第每一行的数据与其他行的数据进行矩阵乘法开根号得到两点之间的距离
            dists[i,j]=np.sqrt(np.dot((vi-vj),(vi-vj)))
    return dists#返回一个矩阵
def find_near_pionts(point_id,eps,dists):
    #得到point_id周围的点和它的距离 如果周围的点距离小于该点围起来的eps 那么返回true说明是该点的临近点
    index=(dists[point_id]<=eps)
    return np.where(True==index)[0].tolist()#tolist()是将矩阵转换成列表

def expand_cluster(dists,labs,cluster_id,seeds,eps,min_points):
    #获取一个临近点 对该点进行处理
    i=0
    #遍历到该点大于seeds的点数为止 这样就可以保证每个点都被归类到
    while i<len(seeds):
        Pn=seeds[i]
        if labs[Pn]==NOISE:
            labs[Pn]=cluster_id
        elif labs[Pn]==UNCLASSIFIED:
            #把该点归类为当前簇
            labs[Pn]=cluster_id
            #计算该点的临近点 创建一个新的临近点
            new_seeds=find_near_pionts(Pn,eps,dists)
            #让该临近点归类为原来的点
            if len(new_seeds)>=min_points:
                seeds=seeds+new_seeds
            else:
                continue
        #自加 开始下一次循环
        i=i+1


def DBSCAN(datas,eps,min_points):
    #得到每两点之间距离
    dists=getdatadisstance(datas)

    n_points=datas.shape[0]#shape返回第一维度的参数，得到行的数目

    labs=[UNCLASSIFIED]*n_points
    #将全部点设为没有被标记的点（0）
    cluster_id=0#其实簇id为0

    #遍历所有点
    for point_id in range(0,n_points):
        #如果当前点已经被标记 那么跳过这个点 进行下一个点的遍历
        if not(labs[point_id]==UNCLASSIFIED):
            continue
        seeds = find_near_pionts(point_id, eps, dists)  # 得到该点的临近点,用这些点来进行扩展

        #如果临近点周围的点数小于设置的阈值 那么标记其为噪声点
        if len(seeds)<min_points:
            labs[point_id]=NOISE
        else:
            #如果该点没有被标记 簇加1 然后进行扩张
            cluster_id=cluster_id+1
            #标记该点的当前簇值
            labs[point_id]=cluster_id
            expand_cluster(dists,labs,cluster_id,seeds,eps,min_points)
    return labs,cluster_id
            #进行该簇的扩张(聚类扩展)

def draw_cluster(datas,labs,n_cluster):
    plt.cla()#清楚axes 清除指定范围的绘图区域

    #用推导式 为聚类点分配颜色
    colors=[plt.cm.Spectral(each) for each in np.linspace(0,1,n_cluster)]
    for i,lab in enumerate(labs):#i为返回下标的值  lab为labs里面的标记值
        if lab==NOISE:
            plt.scatter(datas[i,0],datas[i,1],s=16,color=(0,0,0))
        else:
            plt.scatter(datas[i,0],datas[i,1],s=16,color=colors[lab-1])
    plt.show()

if __name__=="__main__":
    path = 'dataset/test3.pkl'
    f = open(path, 'rb')
    data = pickle.load(f)
    PCAdata = []
    dataLabels = []
    origin_labels = []
    for jsonData in data:
        PCAdata.append(data[jsonData])
        dataLabels.append(data[jsonData].argmax())
    X = np.array(PCAdata)
    pca = PCA(n_components=2)
    pca.fit(X)
    datas=X
    datas=StandardScaler().fit_transform(datas)
    eps=0.70#设置的种子点的搜索半径
    min_points=5#设置的阈值 大于此阈值既可以标记为一个种子点
    labs,cluster_id=DBSCAN(datas,eps=eps,min_points=min_points)
    draw_cluster(datas,labs,cluster_id)


















