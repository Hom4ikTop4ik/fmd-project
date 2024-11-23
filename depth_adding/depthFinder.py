from pywavefront import Wavefront
import numpy as np
import matplotlib.pyplot as plt


def showCameraImage(coordsDisplay):
    x = []
    z = []
    for vect in coordsDisplay:
        x.append(vect[0])
        z.append(vect[1])
    plt.scatter(x, z, s=3, linewidths=0.1)
    # ax = fig.add_subplot(111)
    # ax.scatter(x, z, c='b', marker='o', linewidths=0.1)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Z')
    # lim = 2
    # ax.set_xlim(-lim,lim)
    # ax.set_ylim(-lim,lim)
    plt.show()
    

def showPointCloud(verts):
    x = []
    y = []
    z = []
    for vect in verts:
        x.append(vect[0])
        y.append(vect[1])
        z.append(vect[2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', marker='o', linewidths=0.1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    lim = 0.5
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_zlim(-lim,lim)
    plt.show()

def calcDisplayCoords(pointVec):
    px, py, pz = pointVec[0], pointVec[1], pointVec[2]
    return np.array((px/py, pz/py))

def calcCameraCoords(angleHor, angleVer, camDist):
    kx = camDist * np.cos(angleVer) * np.cos(angleHor)
    ky = camDist * np.cos(angleVer) * np.sin(angleHor)
    kz = camDist * np.sin(angleVer)
    return kx, ky, kz



def calcCameraBasedCoords(angleHor, angleVer, camDist, pointVec):
    # in terms of bloknot solution angleVer is psi, angleHor is phi
    
    # first calc coords of camera (prefix k for them)
    kx, ky, kz = calcCameraCoords(angleHor, angleVer, camDist)
    px, py, pz = pointVec[0], pointVec[1], pointVec[2]
    
    # find answer vector in initial basis
    xdif = kx - px
    ydif = ky - py 
    zdif = kz - pz 
    difvec = -np.array((xdif, ydif, zdif))

    # find every coordinate of answer vector in camera basis

    # calc X coordinate in camera basis
    eXx = np.sin(angleHor)
    eXy = -np.cos(angleHor)
    eXz = 0
    eX = np.array((eXx, eXy, eXz)) 
    ansX = np.dot(difvec, eX)
    
    # calc Y coordinate in camera basis
    eYx = -np.cos(angleVer) * np.cos(angleHor)
    eYy = -np.cos(angleVer) * np.sin(angleHor)
    eYz = -np.sin(angleVer)
    eY = np.array((eYx, eYy, eYz)) 
    ansY = np.dot(difvec, eY)

    # calc Z coordinate in camera basis
    eZx = -np.sin(angleVer) * np.cos(angleHor)
    eZy = -np.sin(angleVer) * np.sin(angleHor)
    eZz = np.cos(angleVer)
    eZ = np.array((eZx, eZy, eZz)) 
    ansZ = np.dot(difvec, eZ)

    return np.array((ansX, ansY, ansZ))



def standartiseArray(verts):
    ans = []
    for vect in verts:
        ans.append((vect[0],vect[2],vect[1]))
    return ans


def loss(currentRelList, targetRelList):
    # lists of length relations from camera and from image
    l = 0.0
    for curRel, targRel in zip(currentRelList, targetRelList):
        l += ((curRel - targRel) ** 2) 
    return l / len(currentRelList) # TODO correct for the case when relation is very big or very small 

def getRelation3d(id1a, id2a, id1b, id2b, vects, angleHor, angleVer, camDist):
    # calculate relation between lengths of segments A and B on the camera screen
    # segments are given each by indexes of two points, 1 and 2. 
    camDisplay = []
    for vect in [vects[id1a], vects[id2a], vects[id1b], vects[id2b]]:
        v = calcCameraBasedCoords(angleHor, angleVer, camDist, vect)
        camDisplay.append(calcDisplayCoords(v))
    segAlen = np.linalg.norm(camDisplay[0] - camDisplay[1])
    segBlen = np.linalg.norm(camDisplay[2] - camDisplay[3])
    return (segAlen / segBlen)

def getRelation2d(id1a, id2a, id1b, id2b, vects2d):
    segAlen = np.linalg.norm(vects2d[id1a] - vects2d[id2a])
    segBlen = np.linalg.norm(vects2d[id1b] - vects2d[id2b])
    return (segAlen / segBlen)


def findDepth(faceTemplateFileName, coordsFromImageRaw,
              lEyeCornerIdImg,
              rEyeCornerIdImg,
              upNoseIdImg, idMapping:dict,
              accuracy = 40,
              debug:bool = False):

    lEyeCornerIdObj = idMapping[lEyeCornerIdImg]
    rEyeCornerIdObj = idMapping[rEyeCornerIdImg]
    upNoseIdObj = idMapping[upNoseIdImg]         

    face = Wavefront(faceTemplateFileName)
    verts = standartiseArray(face.vertices)

    coordsFromImg = []
    for coord in coordsFromImageRaw:
        coordsFromImg.append(np.array(coord))

    angleHor = np.pi / 2
    angleVer = 0.0
    camDist = 3
    stepSize = 0.01
    if debug:
        pcloud = []
        for key in idMapping.keys():
            pcloud.append(verts[idMapping[key]])
        showPointCloud(pcloud)

    vects = []
    for vert in verts:
        vects.append(np.array(vert))

    for i in range(accuracy):
        # calculate the gradient. 
        # Get loss, then add 0.0001 to angleHot, and get loss again 
        delta = 0.0001
        relCurrent = getRelation3d(lEyeCornerIdObj, upNoseIdObj, rEyeCornerIdObj, upNoseIdObj, vects,
                            angleHor + delta, angleVer, camDist)
        relTarget = getRelation2d(lEyeCornerIdImg, upNoseIdImg, rEyeCornerIdImg, upNoseIdImg, coordsFromImg)
        loss1 = loss([relCurrent], [relTarget])
        relCurrent = getRelation3d(lEyeCornerIdObj, upNoseIdObj, rEyeCornerIdObj, upNoseIdObj, vects,
                            angleHor - delta, angleVer, camDist)
        loss2 = loss([relCurrent], [relTarget])
        horizPartDeriv = (loss1 - loss2) / delta # partial derivative of loss on horizontal angle
        if debug:
            print(loss1, loss2, horizPartDeriv)
        angleHor -= horizPartDeriv * stepSize # TODO add partial derivative for the angleVer 

    if debug:
        camDisplay = []
        for key in idMapping.keys():
            disvec = calcCameraBasedCoords(angleHor, angleVer, camDist, vects[idMapping[key]])
            if debug:
                camDisplay.append(calcDisplayCoords(disvec))
        if debug:
            print(angleHor)
            showCameraImage(camDisplay)

    camvec = calcCameraCoords(angleHor, angleVer, camDist)
    if debug:
        print("id mapping keys: ", idMapping.keys())
    depthList = []
    for i in range(len(idMapping.keys())):
        pointVec = vects[idMapping[i]]
        depth = np.linalg.norm(camvec - pointVec)
        depthList.append(depth)
    return depthList