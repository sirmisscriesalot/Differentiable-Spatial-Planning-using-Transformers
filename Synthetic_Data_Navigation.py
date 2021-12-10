from random import randint,uniform,choice
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon,Point
from shapely import affinity
import heapq
import math
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--size",help = "size of the dataset",type=int)
parser.add_argument("--M",help= "size of the map",type=int)
parser.add_argument("--type",help="name the type of set you're generating")
args = parser.parse_args()

name = args.type

# M = 15
M = args.M

def createMapGoal():
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

def createNodes():
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

def getNeighbor(m,point):
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

def Dijkstra(m,goal):
  Q = []
  nodes = createNodes()
  nodes[goal[0]][goal[1]].g = 0
  heapq.heappush(Q,nodes[goal[0]][goal[1]])

  while len(Q)>0:
    current = heapq.heappop(Q)
    for i in getNeighbor(m,current.position):
      temp = current.g + 1
      if temp < nodes[i[0]][i[1]].g:
        nodes[i[0]][i[1]].g = temp
        nodes[i[0]][i[1]].parent = current

        if not nodes[i[0]][i[1]] in Q:
          heapq.heappush(Q,nodes[i[0]][i[1]])

  return nodes

def getOutput(nodes):
  y = np.zeros((M,M))
  for i in nodes:
    for j in i:
      y[j.position] = j.g

  for i in range(M):
    for j in range(M):
      if y[i,j] == math.inf:
        y[i,j] = -1

  return y

def createOutputVisualization(nodes):
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

def createData():
  m,g,goalcoord = createMapGoal()
  x = np.stack((m,g)) #creating the input 

  n = Dijkstra(m,goalcoord)
  y = getOutput(n)

  return x,y


input_file_name = "navdata_input_" + str(args.M) + "_" + str(args.size) + "_" + name +".npy"
output_file_name = "navdata_output_" + str(args.M) + "_" + str(args.size) + "_" + name +".npy"
with open(input_file_name,'wb') as f1, open (output_file_name,'wb') as f2:
  for i in tqdm(range(args.size),desc="creating dataset..." ):
    input,output = createData()
    np.save(f1,input)
    np.save(f2,output)

# with open('nav_data_input.npy','rb') as f1, open ('nav_data_output.npy','rb') as f2:
#   for i in range(10):
#     a = np.load(f1)
#     b = np.load(f2)
#     createMapGoalVisualization(a[0],a[1])
    
#     for i in range(M):
#       for j in range(M)
#         if b[i,j] == -1:
#           b[i,j] = math.inf

#     plt.subplot(133)
#     plt.imshow(b,cmap='viridis')

#     plt.show()

#it seems to be working 

#should probably also add visualization tools separately ?? 
    
