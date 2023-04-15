import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path_dtlane = "230415/dtlane.csv"
path_lane = "230415/lane.csv"
path_line = "230415/line.csv"
path_node = "230415/node.csv"
path_point = "230415/point.csv"
path_white_line = "230415/whiteline.csv"

# DID,Dist,PID,Dir,Apara,r,slope,cant,LW,RW
dtlane = pd.read_csv(path_dtlane, header=0, sep=",")
dtlane = np.array(dtlane)

# LnID,DID,BLID,FLID,BNID,FNID,JCT,BLID2,BLID3,BLID4,FLID2,FLID3,FLID4,ClossID,Span,LCnt,Lno,LaneType,LimitVel,RefVel,RoadSecID,LaneChgFG
lane = pd.read_csv(path_lane, header=0, sep=",")
lane = np.array(lane)

# LID,BPID,FPID,BLID,FLID
line = pd.read_csv(path_line, header=0, sep=",")
line = np.array(line)

# NID,PID
node = pd.read_csv(path_node, header=0, sep=",")
node = np.array(node)

# PID,B,L,H,Bx,Ly,ReF,MCODE1,MCODE2,MCODE3
points = pd.read_csv(path_point, header=0, sep=",")
points = np.array(points)

# ID,LID,Width,Color,type,LinkID
white_line = pd.read_csv(path_white_line, header=0, sep=",")
white_line = np.array(white_line)

# 提取WhiteLine
# LID,BPIDx,BPIDy,FPIDx,FPIDy
BPID, FPID = 0, line[0, 1]
free_point = 0
free_point_pre = 2
WhiteLine = np.zeros([white_line.shape[0] + free_point_pre, 5])
for i in range(white_line.shape[0]):
    LID = white_line[i, 1]
    if line[line[:, 0] == LID, 1] != FPID:
        BPID, FPID = FPID, BPID
        free_point = free_point + 1
        WhiteLine[-free_point, 0] = LID - 1
        WhiteLine[-free_point, 1:3] = np.array([points[points[:, 0] == BPID, 4], points[points[:, 0] == BPID, 5]]).T
        WhiteLine[-free_point, 3:5] = np.array([points[points[:, 0] == FPID, 4], points[points[:, 0] == FPID, 5]]).T
    BPID = line[line[:, 0] == LID, 1]
    FPID = line[line[:, 0] == LID, 2]
    WhiteLine[i, 0] = LID
    WhiteLine[i, 1:3] = np.array([points[points[:, 0] == BPID, 4], points[points[:, 0] == BPID, 5]]).T
    WhiteLine[i, 3:5] = np.array([points[points[:, 0] == FPID, 4], points[points[:, 0] == FPID, 5]]).T

if free_point != free_point_pre:
    print(f"WhiteLine自由点数量不对，当前free_point={free_point}，与free_point={free_point_pre}不符合")
    raise IndexError

# 提取Lane
# LnID,BNIDx,BNIDy,FNIDx,FNIDy,Lno
free_point = 0
free_point_pre = 2
BNID, FNID = 0, lane[0, 4]
Lno = 0
Lane = np.zeros([lane.shape[0] + free_point_pre, 6])
for i in range(lane.shape[0]):
    LnID = lane[i, 0]
    if lane[i, 4] != FNID:
        BNID, FNID = FNID, BNID
        BPID = node[node[:, 0] == BNID, 1]
        FPID = node[node[:, 0] == FNID, 1]
        free_point = free_point + 1
        Lane[-free_point, 0] = LnID - 1
        Lane[-free_point, 1:3] = np.array([points[points[:, 0] == BPID, 4], points[points[:, 0] == BPID, 5]]).T
        Lane[-free_point, 3:5] = np.array([points[points[:, 0] == FPID, 4], points[points[:, 0] == FPID, 5]]).T
        Lane[-free_point, 5] = Lno
    BNID = lane[i, 4]
    FNID = lane[i, 5]
    BPID = node[node[:, 0] == BNID, 1]
    FPID = node[node[:, 0] == FNID, 1]
    Lno = lane[i, 16]
    Lane[i, 0] = LnID
    Lane[i, 1:3] = np.array([points[points[:, 0] == BPID, 4], points[points[:, 0] == BPID, 5]]).T
    Lane[i, 3:5] = np.array([points[points[:, 0] == FPID, 4], points[points[:, 0] == FPID, 5]]).T
    Lane[i, 5] = Lno
LnID = lane[-1, 0]
BNID, FNID = FNID, BNID
BPID = node[node[:, 0] == BNID, 1]
FPID = node[node[:, 0] == FNID, 1]
free_point = free_point + 1
Lane[-free_point, 0] = LnID - 1
Lane[-free_point, 1:3] = np.array([points[points[:, 0] == BPID, 4], points[points[:, 0] == BPID, 5]]).T
Lane[-free_point, 3:5] = np.array([points[points[:, 0] == FPID, 4], points[points[:, 0] == FPID, 5]]).T
Lane[-free_point, 5] = Lno

if free_point != free_point_pre:
    print(f"Lane的自由点数量不对，当前free_point={free_point}，与free_point={free_point_pre}不符合")
    raise IndexError

np.save("230415/data/white_line.npy", WhiteLine)
np.save("230415/data/lane.npy", Lane)

# plt.figure()
# plt.plot(-points[:, 4], points[:, 5], ".", color='r')
# plt.plot(-WhiteLine[:, 1], WhiteLine[:, 2], "*", color='b')
# plt.plot(-Lane[Lane[:, 5] == 1, 1], Lane[Lane[:, 5] == 1, 2], "*", color="g")
# plt.show()



