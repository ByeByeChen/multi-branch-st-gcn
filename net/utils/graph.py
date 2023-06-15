import numpy as np

class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                        11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'rat':
            self.num_node = 12
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (0, 2), (0, 3), (3, 6), (4, 6), (5,6),
                             (6, 7), (6, 8), (6, 9), (9, 10), (10, 11)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        # elif layout=='customer settings'
        #     pass
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        # self.dilation = 1 self.max_hop = 1
        # 其中dilation=1 表示只考虑相连的节点
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        # 得到一个邻接矩阵 相连的节点为1 root节点也为1 和 hop_dis的区别就在 root节点的值 还有剩下的节点值为0 hop_dis中为inf
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
            #  这里是做矩阵的归一化也就是用度矩阵做归一化
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            # 这个划分策略表示Uni-labeling
            # partitioning strategy, where all nodes in a neighborhood has the same label
            # 根据论文中所述：feature vectors on every neighboring node will have a inner product with the same weight vector
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
            print("use_strategy:","uniform")
        elif strategy == 'distance':
            # 这个就是distance partitioning
            # 将节点分成两部分
            # where d = 0 refers to the root node itself and
            # remaining neighbor nodes are in the d = 1 subset.
            # shape (2, num_node, num_node)
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                # hop == 0 : 从hop_dis中取出节点指等于0的赋值  也就是root 对应root node it self
                # hop == 1 : 从hop_dis中取出节点值等于1的赋值 也就是neighbor node 相连的节点
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
            print("use_strategy:", "distance")
        elif strategy == 'spatial':
            # 最后一个空间划分策略
            # 将节点 分成三部分
            # 1) the root node itself;
            # 2)centripetal group: the neighboring nodes
            # that are closer to the gravity center of the skeleton than the root node;
            # 3) otherwise the centrifugal group
            # 这里用一个数组存储
            A = []
            for hop in valid_hop:
                # root node
                a_root = np.zeros((self.num_node, self.num_node))
                # the neighboring nodes that are closer to the gravity center
                a_close = np.zeros((self.num_node, self.num_node))
                # otherwise the centrifugal group
                a_further = np.zeros((self.num_node, self.num_node))
                # 下面分析怎么实现的
                # 0 if rj = ri
                # 1 if rj < ri
                # 2 if rj > ri
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        # 这个if 表示取出有效值 hop_dis中的 0, 1 也就是有边链接关系的节点包括root node itself
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                # 这里就是root节点赋值
                                # 当hop == 0 时 进入此if的都是root itself i == j 表示根节点
                                # hop == 1 时 进入这里的表示 i, j 有连接 但是和center没有连接 inf
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                # the neighboring nodes that are closer to the gravity center
                                # 表示 i 到center的距离比 j 到center的距离近
                                # hop == 1 进入此条件语句
                                # 当 hop_dis[j, self.center] == inf hop_dis[i, self.center] == 1.0
                                # 或者 hop_dis[j, self.center] == 1.0 hop_dis[i, self.center] == 0.0 都可以进入此条件语句
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                # otherwise the centrifugal group
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            # 最终拼成一个三维矩阵当作权重输入模型
            # shape (3, num_node, num_node)
            # A[0] 有root节点还有和center相连的节点赋予权重值（也就是距离值）
            # A[1] （a_root + a_close）在A[0]上增加了比root距离中心点近的权重值
            # A[2] 就是比root距离中心点远的权重值
            self.A = A
            print("use_strategy:", "spatial")
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD