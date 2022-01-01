from random import randint,uniform,choice
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon,Point
from shapely import affinity
import heapq
import math
import argparse
from tqdm import tqdm
import time 
import concurrent.futures

t1 = time.perf_counter()

parser = argparse.ArgumentParser()
parser.add_argument("--size",help = "size of the dataset",type=int)
parser.add_argument("--M",help= "size of the map",type=int)
parser.add_argument("--xfile",help="name the of input x dataset you're generating")
parser.add_argument("--yfile",help="name the of output y dataset you're generating")
parser.add_argument("--mode",help="enter c to create dataset and v to visualize the created dataset")
parser.add_argument("--nthread",help="number of threads you want to use",type=int)
args = parser.parse_args()

map_size = args.M
mode = args.mode
xfile = args.xfile
yfile = args.yfile
nthread = args.nthread
size = args.size
iter = range(size)

# per_thread_queue = []
# for i in range(nthread):
#   per_thread_queue.append(size//nthread)
# per_thread_queue[-1] = per_thread_queue[-1] + size%nthread

# M = 15
# M = args.M

def createMapGoal(M):
  O = randint(0,5)
  m = np.zeros((M,M))
  g = np.zeros((M,M))

  bounding_m = Polygon([(0,0),(0,M),(M,M),(M,0)])

  for i in range(O):
    rec_length = randint(1,int(M/2))
    rec_breadth = randint(1,int(M/2))

    center_x = randint(0,M)
    center_y = randint(0,M)

    co_ord1 = (center_x - rec_length/2,center_y - rec_breadth/2)
    co_ord2 = (center_x + rec_length/2,center_y - rec_breadth/2)
    co_ord3 = (center_x + rec_length/2,center_y + rec_breadth/2)
    co_ord4 = (center_x - rec_length/2,center_y + rec_breadth/2)

    bounding_o = Polygon([co_ord1,co_ord2,co_ord3,co_ord4])
    angle = randint(0,360)
    bounding_o_rot = affinity.rotate(bounding_o,angle,(center_x,center_y)) #rotating a single time here , not really sure what they mean by rotating twice
    
    o_map = bounding_m.intersection(bounding_o_rot)

    for row in range(M):
      for col in range(M):
        point = Point(row,col)
        if o_map.contains(point):
          m[row,col] = 1

  free_space = []

  for row in range(M):
    for col in range(M):
      if m[row,col]==0:
        free_space.append((row,col))

  ran = free_space[randint(0,len(free_space)-1)]
  g[ran] = 1

  return m,g,ran

def createMapGoalVisualization(m,g):
  plt.figure(figsize=(9, 3))

  plt.subplot(131)
  plt.imshow(m,cmap='binary')

  plt.subplot(132)
  plt.imshow(g,cmap='binary')

class Node():
    def __init__(self, parent=None, position=None,g=None):
        self.parent = parent
        self.position = position
        self.g = g

    def __lt__(self, other):
        return self.g < other.g

def createNodes(M):
  node_list = []
  for i in range(M):
    node_list.append([])
    for j in range(M):
      node_list[i].append(Node(parent=None,position=(i,j),g=math.inf)) 

  return node_list 

def isBlocked(m,point):
  if m[point]==1:
    return True
  return False

def getNeighbor(m,point,M):
  x,y = point 
  neighbor = []
  
  if x > 0:
    neighbor.append((x-1,y))
  if x < M-1:
    neighbor.append((x+1,y))
  if y > 0:
    neighbor.append((x,y-1))
  if y < M-1:
    neighbor.append((x,y+1))

  points = []
  for i in neighbor:
    if not isBlocked(m,i):
      points.append(i)
  
  return points

def Dijkstra(m,goal,M):
  Q = []
  nodes = createNodes(M)
  nodes[goal[0]][goal[1]].g = 0
  heapq.heappush(Q,nodes[goal[0]][goal[1]])

  while len(Q)>0:
    current = heapq.heappop(Q)
    for i in getNeighbor(m,current.position,M):
      temp = current.g + 1
      if temp < nodes[i[0]][i[1]].g:
        nodes[i[0]][i[1]].g = temp
        nodes[i[0]][i[1]].parent = current

        if not nodes[i[0]][i[1]] in Q:
          heapq.heappush(Q,nodes[i[0]][i[1]])

  return nodes

def getOutput(nodes,M):
  y = np.zeros((M,M))
  for i in nodes:
    for j in i:
      y[j.position] = j.g

  for i in range(M):
    for j in range(M):
      if y[i,j] == math.inf:
        y[i,j] = -1

  return y

def createOutputVisualization(nodes,M):
  y = np.zeros((M,M))
  for i in nodes:
    for j in i:
      y[j.position] = j.g

  plt.subplot(133)
  plt.imshow(y,cmap='viridis')

# m,g,goalcoord = createMapGoal()
# x = np.stack((m,g)) #creating the input 
# createMapGoalVisualization(m,g)

# n = Dijkstra(m,goalcoord)
# y = getOutput(n)
# createOutputVisualization(n)

# plt.show()

def createData(_):
  M = map_size
  m,g,goalcoord = createMapGoal(M)
  x = np.stack((m,g)) #creating the input 

  n = Dijkstra(m,goalcoord,M)
  y = getOutput(n,M)

  return (x,y)

def close_event():
  plt.close()

# def createDataThread(tempsize):
#   inout_list = []
#   for i in range(tempsize):
#     temp_in,temp_out = createData(map_sizes[i%3])
#     inout_list.append((temp_in,temp_out))

#   return inout_list


# input_file_name = "navdata_input_" + str(args.M) + "_" + str(args.size) + "_" + name +".npy"
# output_file_name = "navdata_output_" + str(args.M) + "_" + str(args.size) + "_" + name +".npy"
# input_file_name = "navdata_input_" + str(args.size) + "_" + name +".npy"
# output_file_name = "navdata_output_" + str(args.size) + "_" + name +".npy"

input_file_name = xfile +".npz"
output_file_name = yfile +".npz"

def main():
  if mode=='c':
    # with open(input_file_name,'wb') as f1, open (output_file_name,'wb') as f2:
    #   for i in tqdm(range(args.size),desc="creating dataset..." ):
    #     input,output = createData(map_sizes[i%3])
    #     np.save(f1,input)
    #     np.save(f2,output)
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #   results = executor.map(createDataThread,per_thread_queue)

    #   with open(input_file_name,'wb') as f1, open (output_file_name,'wb') as f2:
    #     for result in results:
    #       for data in result:
    #         input = data[0]
    #         output = data[1]

    #         np.save(f1,input)
    #         np.save(f2,output)
    with concurrent.futures.ProcessPoolExecutor(max_workers=nthread) as executor:
      futures = list(tqdm(executor.map(createData,iter),total=len(iter)))
    #   print(len(results))
    #   with open(input_file_name,'wb') as f1, open (output_file_name,'wb') as f2:
    #     for result in results:
    #       input = result[0]
    #       output = result[1]

    #       np.save(f1,input)
    #       np.save(f2,output)       
    #   futures = [executor.submit(createData,i) for i in iter]
    #   concurrent.futures.wait(futures,timeout=10000)
    #   np.savez(input_file_name,*[i.result()[0] for i in futures])
    #   np.savez(output_file_name,*[i.result()[1] for i in futures])
      np.savez(input_file_name,*[i[0] for i in futures])
      np.savez(output_file_name,*[i[1] for i in futures])

#   check = np.load(input_file_name)
#   print(len(check.files))

  fig = plt.figure()
  timer = fig.canvas.new_timer(interval = 5000) #creating a timer object and setting an interval of 5000 milliseconds
  timer.add_callback(close_event)

  if mode=='v':
    # with open(input_file_name,'rb') as f1, open (output_file_name,'rb') as f2:
    #   for i in range(10):
    #     a = np.load(f1)
    #     b = np.load(f2)
    #     createMapGoalVisualization(a[0],a[1])
        
    #     for i in range(len(a[0])):
    #       for j in range(len(a[0])):
    #         if b[i,j] == -1:
    #           b[i,j] = math.inf

    #     plt.subplot(133)
    #     plt.imshow(b,cmap='viridis')
    #     timer.start()
    #     plt.show()

    with open(input_file_name,'rb') as f1, open (output_file_name,'rb') as f2:
      inx = np.load(f1)
      outy = np.load(f2)
      for i in range(10):
        a = inx['arr_' + str(i)]
        b = outy['arr_' + str(i)]
        # a = np.load(f1)
        # b = np.load(f2)
        createMapGoalVisualization(a[0],a[1])
        
        for i in range(len(a[0])):
          for j in range(len(a[0])):
            if b[i,j] == -1:
              b[i,j] = math.inf

        plt.subplot(133)
        plt.imshow(b,cmap='viridis')
        timer.start()
        plt.show()

  #it seems to be working 

  #should probably also add visualization tools separately ?? 
  t2 = time.perf_counter()
  print(f'Finished in {t2-t1} seconds')

if __name__ == '__main__':
  main()
