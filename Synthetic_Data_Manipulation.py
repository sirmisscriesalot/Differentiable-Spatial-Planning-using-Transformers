from random import randint,uniform,choice
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon,Point,LineString
from shapely import affinity
import heapq
import math

P = 200
# M = 18
M = 36

def getDist(point1,point2):
  dist = math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
  return dist

def isValidDistance(point1,point2):
  dist = getDist(point1,point2)
  if dist >= 0.25*P or dist <= 0.75*P:
    return True
  return False

def getValidObstacleCenters(point):
  valid_obs_centers = []
  for i in range(P):
    for j in range(P):
      if isValidDistance((i,j),point):
        valid_obs_centers.append((i,j))

  return valid_obs_centers

def createOperationalSpace():
  O = randint(0,5)
  op_space = np.zeros((P,P))
  bounding_p = Polygon([(0,0),(0,P),(P,P),(P,0)])

  center = (P/2,P/2)
  obs_list = []

  for i in range(O):
    obs_cen = choice(getValidObstacleCenters(center))
    D = getDist(obs_cen,center)

    obs_cen = Point(obs_cen)
    obs_r = uniform(0.05*P,D - 0.15*P)
    obs = obs_cen.buffer(obs_r)
    
    o_map = bounding_p.intersection(obs)
    obs_list.append(o_map)

    for row in range(P):
      for col in range(P):
        point = Point(row,col)
        if o_map.contains(point):
          op_space[row,col] = 1

  #right now I assume op_space will only really be useful for visualization 
  #definitely should shift it to a different function don't wanna deal with more overhead
  return op_space,obs_list

def isCollide(obstacles,angle1,angle2):
  center = Point((P/2,P/2))
  link_len = P/4
  end1 = Point((P/2 + link_len,P/2))

  link1 = LineString([center,end1])
  link1 = affinity.rotate(link1,angle1,origin=center,use_radians=True)

  start1 = list(link1.coords)[1]
  start2 = Point(start1)
  end2 = Point((start1[0]+link_len,start1[1]))

  link2 = LineString([start1,end2])
  link2 = affinity.rotate(link2,angle2,origin=start2,use_radians=True)

  for obs in obstacles:
    if obs.intersects(link1) or obs.intersects(link2):
      return True 

  return False

def createMapGoal(obs):
  m = np.zeros((M,M))
  g = np.zeros((M,M))

  for i in range(M):
    for j in range(M):
      if isCollide(obs,2*math.pi*i/M,2*math.pi*j/M):
        m[i,j] = 1

  free_space = []

  for row in range(M):
    for col in range(M):
      if m[row,col]==0:
        free_space.append((row,col))

  ran = free_space[randint(0,len(free_space)-1)]
  g[ran] = 1

  return m,g,ran

def createMapGoalVisualization(m,g,o):
  plt.figure(figsize=(9, 3))

  plt.subplot(141)
  plt.imshow(o,cmap='binary')

  plt.subplot(142)
  plt.imshow(m,cmap='binary')

  plt.subplot(143)
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
  
#   if x > 0:
#     neighbor.append((x-1,y))
#   if x < M-1:
#     neighbor.append((x+1,y))
#   if y > 0:
#     neighbor.append((x,y-1))
#   if y < M-1:
#     neighbor.append((x,y+1))

  neighbor.append(((x-1)%(M),y))
  neighbor.append(((x+1)%(M),y))
  neighbor.append((x,(y-1)%(M)))
  neighbor.append((x,(y+1)%(M)))  

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

  plt.subplot(144)
  plt.imshow(y,cmap='viridis')

vi,obs1 = createOperationalSpace()
m,g,goalcoord = createMapGoal(obs1)
x = np.stack((m,g)) #creating the input 
createMapGoalVisualization(m,g,vi)

n = Dijkstra(m,goalcoord)
y = getOutput(n)
createOutputVisualization(n)

plt.show()