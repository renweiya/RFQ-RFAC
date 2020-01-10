from keras.utils import to_categorical
import numpy as np
from sklearn import neighbors

def adjacency_matrix(state,neighbor_num):
    position = []
    N=len(state)
    for i in range(N):
        position.append([state[i][-2],state[i][-1]])
    position_=np.array(position)
    
    neighbor_num=min(neighbor_num,len(position_))
    tree = neighbors.KDTree(position_, leaf_size=2)         
    _, ind = tree.query(position_, k=neighbor_num)  
    #print(ind)  # indices of neighbor_num closest neighbors
    #print(dist)  # distances to neighbor_num closest neighbors
    adj_matrix=np.zeros((N,N))
    for i in range(N):
        one_hot_neighbors = to_categorical(ind[i,:],num_classes=N)#one_hot representation
        adj_matrix[i,:]=sum(one_hot_neighbors)
  
    return adj_matrix
