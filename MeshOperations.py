import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, det
from scipy.sparse import spdiags, csc_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

class Mesh():
    min: np.ndarray
    max: np.ndarray
    nbNod: int
    POS: np.array
    nbElem: int
    ELE_INFOS: np.ndarray
    nbPoints: int
    nbLines: int
    nbTriangles: int
    nbQuads: int
    nbTets: int
    nbHexas: int
    nbPrisms: int
    nbPyramids: int
    nbLines3: int
    nbTriangles6: int
    POINTS: np.ndarray
    LINES: np.ndarray
    TRIANGLES: np.ndarray
    QUADS: np.ndarray
    PRISMS: np.ndarray
    PYRAMIDS: np.ndarray
    LINES3: np.ndarray
    TRIANGLES6: np.ndarray


class MeshOperations():
    ####################################################################
    #MeshOperations Mesh Wrapper Class
    #   Contains the struct with mesh information by getting the mesh path
    #   and provides element transformation and integration rules
    #   for specific element geometries
    #   All procedures that are specific to the element geometry but not
    #   to the shape functions are gathered here
    ####################################################################
    mesh: Mesh
    

        
    def __init__(self, meshPath = None): 
        self.mesh = self.load_gmsh(meshPath)
        #creating low order data out of second order mesh
        if (self.mesh.nbTriangles6 > 0):
            self.mesh.nbTriangles = self.mesh.nbTriangles6
            self.mesh.nbLines = self.mesh.nbLines3
            self.mesh.TRIANGLES = np.zeros((self.mesh.nbTriangles6,4), dtype=int)
            #first three points of triangle6 are low order points
            ##(corners of the triangle)
            self.mesh.TRIANGLES[0:self.mesh.nbTriangles6 ,0:3] = self.mesh.TRIANGLES6[0:self.mesh.nbTriangles6,0:3]
            self.mesh.TRIANGLES[0:self.mesh.nbTriangles6, 3] = self.mesh.TRIANGLES6[0:self.mesh.nbTriangles6, 6]
            self.mesh.LINES = np.zeros((self.mesh.nbLines3, 3), dtype=int)
            #first two points of triangle3 are low order points
            #(bounds of the line)
            self.mesh.LINES[0:self.mesh.nbLines3,0:2] = self.mesh.LINES3[0:self.mesh.nbLines3,0:2]
            self.mesh.LINES[0:self.mesh.nbLines3, 2] = self.mesh.LINES3[0:self.mesh.nbLines3,3]
        
        return
        
        
    def getMeshInformation(self): 
        print('################################################ \n' % ())
        print('## number of volume elements (triangles): %d \n' % (self.mesh.nbTriangles))
        print('## number of boundary elements (lines): %d \n' % (self.mesh.nbLines))
        print('################################################ \n' % ())
        return
        
        
    def reset(self ,meshPath = None): 
        self.mesh = self.load_gmsh(meshPath)
        return
        
        
    def plot(self ,u = None,component = None): 
        triangles = self.getVolumeElementListTriangles()
        nodes = self.getNodeList()
        nbNodes = self.getNumberNodes()
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='3d')
        u = u.reshape(-1)
        zaxis = u[((component - 1) * nbNodes + 1)-1:(component * nbNodes+1)]
        ax.plot_trisurf(nodes[:,0], nodes[:,1], zaxis, triangles=triangles, cmap='viridis')
        title = 'Solution u_'+str(component)
        fig.suptitle(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return

    def get_plot_data(self ,u = None,component = None): 
        triangles = self.getVolumeElementListTriangles()
        nodes = self.getNodeList()
        nbNodes = self.getNumberNodes()
        u = u.reshape(-1)
        zaxis = u[((component - 1) * nbNodes + 1)-1:(component * nbNodes+1)]
        # ax.set_axis_off()
        return nodes[:,0], nodes[:,1], zaxis, triangles

        
    def getNumberOfLines(self): 
        numElems = self.mesh.nbLines
        return numElems
        
        
    def getNumberOfTriangles(self): 
        numElems = self.mesh.nbTriangles
        return numElems
        
        
    def getTagOfLine(self ,elementNumber = None): 
        #delivers "physical" tag
        tag = self.mesh.LINES[elementNumber,2]
        return tag
        
        
    def getTagOfTriangle(self ,elementNumber = None): 
        #delivers "physical" tag
        tag = self.mesh.TRIANGLES[elementNumber,4]
        return tag
        
        
    def getNumberNodes(self): 
        numNodes = self.mesh.nbNod
        return numNodes
        
        
    def getVolumeElementListTriangles(self): 
        #number of elements
        nbElems = self.getNumberOfTriangles()
        #tags are not included
        volElems = self.mesh.TRIANGLES[0:nbElems,0:3]
        return volElems
        
        
    def getNodeList(self): 
        #only 2D-Nodes
        nodes = self.mesh.POS[:,0:2]
        return nodes
        
        
    def getNodeNumbersOfLine(self ,elementNumber = None,order = None): 
        if (order == 1):
            nodeData = self.mesh.LINES[elementNumber,0:2]
        elif (order == 2):
            nodeData = self.mesh.LINES3[elementNumber,0:3]
        
        return nodeData
        
        
    def getNodeNumbersOfTriangle(self ,elementNumber = None,order = None): 
        if (order == 1):
            nodeData = self.mesh.TRIANGLES[elementNumber,0:3]
        elif (order == 2):
            nodeData = self.mesh.TRIANGLES6[elementNumber,0:6]
        
        return nodeData
        
        
    def calcJacobianOfLine(self, elementNumber = None): 
        #identify nodes belonging to the line segment
        pointNumbers = self.mesh.LINES[elementNumber,0:2]
        points = np.zeros((2,2))
        #reading x- and y-coordinates
        points[:,:] = self.mesh.POS[pointNumbers,0:2]
        #Compute entries of Jacobian (just Vector!!!)
        jacobian = np.zeros((2,1))
        jacobian[0] = 0.5 * (points[1,0] - points[0,0])
        
        jacobian[1] = 0.5 * (points[1,1] - points[0,1])
        
        return jacobian
        
        
    def calcJacobianOfTriangle(self,elementNumber = None): 
        #identify nodes belonging to the triangle
        pointNumbers = self.mesh.TRIANGLES[elementNumber,0:3]
        points = np.zeros((3,2))
        #reading x- and y-coordinates
        points[:,:] = self.mesh.POS[pointNumbers,0:2]
        #Compute entries of Jacobian
        jacobian = np.zeros((2,2))
        jacobian[0,0] = points[1,0] - points[0,0]
        
        jacobian[0,1] = points[1,1] - points[0,1]
        
        jacobian[1,0] = points[2,0] - points[0,0]
        
        jacobian[1,1] = points[2,1] - points[0,1]
        
        return jacobian
        
        
    def calcInverseJacobianOfTriangle(self, elementNumber = None): 
        jacMat = self.calcJacobianOfTriangle(elementNumber)
        inverseJacobian = inv(jacMat)
        return inverseJacobian
        
        
    def calcJacobianDeterminantOfLine(self,elementNumber = None): 
        jacMat = self.calcJacobianOfLine(elementNumber)
        jacobianDet = np.sqrt((jacMat[0,0]) ** 2 + (jacMat[1,0]) ** 2)
        return jacobianDet
        
        
    def calcJacobianDeterminantOfTriangle(self,elementNumber = None): 
        jacMat = self.calcJacobianOfTriangle(elementNumber)
        jacobianDet = np.abs(det(jacMat))
        return jacobianDet
        
        
    def calcMappedIntegrationPointOfLine(self, elementNumber = None,IP = None): 
        jacMat = self.calcJacobianOfLine(elementNumber)
        pointNumbers = self.mesh.LINES[elementNumber,0:2]
        points = np.zeros((2,2))
        #reading x- and y-coordinates
        points[:,:] = self.mesh.POS[pointNumbers,0:2]
        #different reference position due to non unit reference
    #interval
        mappedIP = IP[0] * jacMat + 0.5 * np.array[[[points[1,0] + points[0,0]],[points[1,1] + points[0,1]]]]
        return mappedIP
        
        
    def calcMappedIntegrationPointOfTriangle(self ,elementNumber = None,IP = None): 
        jacMat = self.calcJacobianOfTriangle(elementNumber)
        point1Number = self.mesh.TRIANGLES[elementNumber,0]
        refPos = self.mesh.POS[point1Number,0:2]
        transMat = np.transpose(jacMat)
        tempVec = np.array([np.dot(transMat[0,:],IP), np.dot(transMat[1,:],IP)])
        mappedIP = tempVec + refPos
        return mappedIP
        
        #look out for orientation during geometry setup
        #counter clockwise for outward facing normal
        #clockwise for inside facing normal
        
    def getNormalVectorOfLine(self,elementNumber = None): 
        pointNumbers = self.mesh.LINES[elementNumber,0:2]
        points = np.zeros((2,2))
        #reading x- and y-coordinates
        points[:,:] = self.mesh.POS[pointNumbers,0:2]
        #check direction here
        tangentVec = points[1,:] - points[0,:]
        magnitude = np.sqrt((tangentVec[0]) ** 2 + (tangentVec[1]) ** 2)
        normalVec = np.zeros(2)
        normalVec[0] = tangentVec[1] / magnitude
        normalVec[1] = - tangentVec[0] / magnitude
        return normalVec
        
        
    def L2errorOfPoissonProblem(self ,u = None,order = None): 
        nTriangles = self.getNumberOfTriangles()
        l2Error = 0
        # Integration rule
        quadWeights,quadPoints,numIPs = self.IntegrationRuleOfTriangle()
        # Loop over each triangle
        for i in np.arange(0,nTriangles):
            # Loop over element integration points
            determinant_of_Jacobian = self.calcJacobianDeterminantOfTriangle(i)
            for j in np.arange(0,numIPs):
                if (order == 1):
                    shape = np.zeros((3,1))
                    shape[0] = 1 - quadPoints[j,0] - quadPoints[j,1]
                    shape[1] = quadPoints[j,0]
                    shape[2] = quadPoints[j,1]
                else:
                    if (order == 2):
                        shape = np.zeros((6,1))
                        shape[0] = (1 - quadPoints(j,0) - quadPoints(j,1)) * (1 - 2 * quadPoints(j,0) - 2 * quadPoints(j,1))
                        shape[1] = quadPoints(j,0) * (2 * quadPoints(j,0) - 1)
                        shape[2] = quadPoints(j,1) * (2 * quadPoints(j,1) - 1)
                        shape[3] = 4 * quadPoints(j,0) * (1 - quadPoints(j,0) - quadPoints(j,1))
                        shape[4] = 4 * quadPoints(j,0) * quadPoints(j,1)
                        shape[5] = 4 * quadPoints(j,1) * (1 - quadPoints(j,0) - quadPoints(j,1))
                # Calculate analytical value
                mappedIP = self.calcMappedIntegrationPointOfTriangle(i,quadPoints[j,:])
                u_exact = 0
                fourierOrder = 30
                for k in np.arange(1,fourierOrder+1):
                    for l in np.arange(1,fourierOrder):
                        if (np.mod(l,2) != 0):
                            coeff = 16 / ((np.pi) ** 4) * 1 / (l ** 3 * (2 * k - 1) + l * (2 * k - 1) ** 3 / 4)
                            u_exact = u_exact + coeff * np.sin((k - 0.5) * np.pi * mappedIP[0]) * np.sin(l * np.pi * mappedIP[1])
                            # coefficient and basisfunction for dirichlet bc(=0) on all faces
                            # if((mod(k,2) ~= 0) && (mod(l,2) ~= 0))  
                            # coeff = 16/((k^2+l^2)*k*l*(pi)^4)                            
                            # u_exact = u_exact + coeff * sin(k*pi*mappedIP(1))*sin(l*pi*mappedIP(2)) 

                # local L2 Error
                conVector = self.getNodeNumbersOfTriangle(i,order)
                errorVal = (np.dot(shape.reshape(-1),u[conVector].reshape(-1)) - u_exact) ** 2
                l2Error = l2Error + quadWeights[j] * determinant_of_Jacobian * errorVal
        
        l2Error = np.sqrt(l2Error)
        return l2Error
        
        
    def IntegrationRuleOfLine(self): 
        #Integration of polynomials up to order 5 on the
    #reference segment (interval [-1,1])
        quadPoints = np.zeros((3,1))
        #xi-values (only one coordinate in reference system)
        quadPoints[0] = - np.sqrt(3 / 5)
        quadPoints[1] = 0
        quadPoints[2] = np.sqrt(3 / 5)
        #weights
        quadWeights = np.zeros((3,1))
        quadWeights[0] = 5 / 9
        quadWeights[1] = 8 / 9
        quadWeights[2] = 5 / 9
        
        numIPs = 3
        return quadWeights,quadPoints,numIPs
        
        
    def IntegrationRuleOfTriangle(self): 
        #Integration of polynomials up to order 5 on the unit
        #triangle (taken from D. Braess, Finite Elemente, Springer Verlag)
        quadPoints = np.zeros((7,2))
        #xi-values
        quadPoints[0,0] = 1 / 3
        quadPoints[1,0] = (6 + np.sqrt(15)) / 21
        quadPoints[2,0] = (9 - 2 * np.sqrt(15)) / 21
        quadPoints[3,0] = (6 + np.sqrt(15)) / 21
        quadPoints[4,0] = (6 - np.sqrt(15)) / 21
        quadPoints[5,0] = (9 + 2 * np.sqrt(15)) / 21
        quadPoints[6,0] = (6 - np.sqrt(15)) / 21
        #eta-values
        quadPoints[0,1] = 1 / 3
        quadPoints[1,1] = (6 + np.sqrt(15)) / 21
        quadPoints[2,1] = (6 + np.sqrt(15)) / 21
        quadPoints[3,1] = (9 - 2 * np.sqrt(15)) / 21
        quadPoints[4,1] = (6 - np.sqrt(15)) / 21
        quadPoints[5,1] = (6 - np.sqrt(15)) / 21
        quadPoints[6,1] = (9 + 2 * np.sqrt(15)) / 21
        #weights
        quadWeights = np.zeros((7,1))
        quadWeights[0] = 9 / 80
        quadWeights[np.arange(1,4)] = (155 + np.sqrt(15)) / 2400
        quadWeights[np.arange(4,7)] = (155 - np.sqrt(15)) / 2400
        
        numIPs = 7
        return quadWeights,quadPoints,numIPs
        
        
    def solve(self, A = None,f = None): 
        #u = np.linalg.solve(A,f)
        #return u
        leaveOutDofs = np.array(np.asarray(np.sum(np.abs(A), axis=0) == 0).nonzero()).reshape(-1)
        if (np.asarray(leaveOutDofs).size == 0):
            u = np.linalg.solve(A,f)
        else:
            initialSize = f.size
            #first elimination of possible zero rows and columns
            A = A[np.any(A, axis=0),:]
            A = A[:,np.any(A,axis=0)]
            #adjusting r.h.s
            f = np.delete(f, leaveOutDofs, None)
            #then solve linear system of equations
            u = np.linalg.solve(A,f)
            #expand solution vector to original size and fill with zeros
    #reduce to true vector
            u = u
            #do insertion
            tempPos = np.zeros(initialSize, dtype=bool)
            tempPos[leaveOutDofs] = True
            tempVec = tempPos.astype(float)
            tempVec[tempPos] = 0
            tempVec[~tempPos] = u
            u = tempVec
        print('solve completed')
        
        return u
        
        
    def load_gmsh(self, filename = None): 
    # Reads a mesh in msh format, version 1 or 2
    # Copyright (C) 10/2007 R Lorph?vre (r.lorphevre@ulg.ac.be)
    # Based on load_gmsh supplied with gmsh-2.0 and load_gmsh2 from JP
    # Moitinho de Almeida (moitinho@civil.ist.utl.pt)
    # number of nodes in function of the element type
        NODES_PER_TYPE_OF_ELEMENT = np.array([2,3,4,4,8,6,5,3,6,9,10,27,18,14,1,8,20,15,13])
        # The format 2 don't sort the elements by reg phys but by
    # reg-elm. If this classification is important for your program,
    # use this (after calling this function):
        
        # [OldRowNumber, NewRowNumber] = sort(OldMatrix(:,SortColumn));
    # NewMatrix = OldMatrix(NewRowNumber,:);
        
        # Change the name of OldMatrix and NewMatrix with the name of yours
    # SortColumn by the number of the last column

        mesh = Mesh()
        mesh.MIN = np.zeros((3,1))
        mesh.MAX = np.zeros((3,1))
        fid = open(filename,'r')
        while 1:
    
            endoffile = 0
            while 1:
    
                tline = fid.readline().rstrip()
                if (tline == ""):
                    endoffile = 1
                    break
                if (tline[0] == '$'):
                    if tline[1] == 'N' and tline[2] == 'O':
                        fileformat = 1
                        break
                    if tline[1] == 'M' and tline[2] == 'e':
                        fileformat = 2
                        tline = fid.readline().rstrip()
                        tline = fid.readline().rstrip()
                        if (tline[0] == '$' and tline[1] == 'E' and tline[2] == 'n'):
                            tline = fid.readline().rstrip()
                            break
                        else:
                            print(' This program can only read ASCII mesh file')
                            print(' of format 1 or 2 from GMSH, try again?')
                            endoffile = 1
                            break
                    if tline[1] == 'E' and (tline[2] == 'L' or tline[2] == 'l'):
                        break
    
            if endoffile == 1:
                break
            if tline[1] == 'N' and (tline[2] == 'O' or tline[2] == 'o'):
                tline = fid.readline().rstrip()
                mesh.nbNod = int(tline) 
                IDS = np.zeros(mesh.nbNod)
                mesh.POS = np.zeros((mesh.nbNod,3))
                for I in np.arange(0,mesh.nbNod).reshape(-1):
                    tline = fid.readline().rstrip()
                    iNod = int(tline.split()[0]) -1
                    X = np.array(tline.split()[1:], dtype=float)
                    IDS[iNod] = I
                    if (I == 1):
                        mesh.MIN = X
                        mesh.MAX = X
                    else:
                        if mesh.MAX[0] < X[0]:
                            mesh.MAX[0] = X[0]
                        if mesh.MAX[1] < X[1]:
                            mesh.MAX[1] = X[1]
                        if mesh.MAX[2] < X[2]:
                            mesh.MAX[2] = X[2]
                        if mesh.MIN[0] > X[0]:
                            mesh.MIN[0] = X[0]
                        if mesh.MIN[1] > X[1]:
                            mesh.MIN[1] = X[1]
                        if mesh.MIN[2] > X[2]:
                            mesh.MIN[2] = X[2]
                    mesh.POS[I,0] = X[0]
                    mesh.POS[I,1] = X[1]
                    mesh.POS[I,2] = X[2]
                tline = fid.readline().rstrip()
            elif tline[1] == 'E' and (tline[2] == 'L' or tline[2] == 'l'):
                tline = fid.readline().rstrip()
                mesh.nbElem = int(tline)
                if (fileformat == 1):
                    nbinfo = 5
                    tags = 2
                if (fileformat == 2):
                    nbinfo = 4
                    tags = 3
                mesh.ELE_INFOS = np.zeros((mesh.nbElem, nbinfo), dtype=int)
                mesh.nbPoints = 0
                mesh.nbLines = 0
                mesh.nbTriangles = 0
                mesh.nbQuads = 0
                mesh.nbTets = 0
                mesh.nbHexas = 0
                mesh.nbPrisms = 0
                mesh.nbPyramids = 0
                #own addition#
                ##############
                mesh.nbLines3 = 0
                mesh.nbTriangles6 = 0
                #############
                mesh.POINTS = np.zeros((mesh.nbElem,2), dtype=int)
                mesh.LINES = np.zeros((mesh.nbElem,3), dtype=int)
                mesh.TRIANGLES = np.zeros((mesh.nbElem,4), dtype=int)
                mesh.QUADS = np.zeros((mesh.nbElem,5), dtype=int)
                mesh.TETS = np.zeros((mesh.nbElem,5), dtype=int)
                mesh.HEXAS = np.zeros((mesh.nbElem,9), dtype=int)
                mesh.PRISMS = np.zeros((mesh.nbElem,7), dtype=int)
                mesh.PYRAMIDS = np.zeros((mesh.nbElem,6), dtype=int)
                #%own addition%
                ##############
                mesh.LINES3 = np.zeros((mesh.nbElem, 4), dtype=int)
                mesh.TRIANGLES6 = np.zeros((mesh.nbElem, 7), dtype=int)
                #############
                for I in np.arange(0,mesh.nbElem):
                    tline = fid.readline().rstrip()

                    
                    mesh.ELE_INFOS[I, :] = np.array(tline.split()[:nbinfo], dtype=int)
                    # decrease element number to get 0-indexing
                    mesh.ELE_INFOS[I,0] -= 1
                    if (fileformat == 1):
                        # take the mesh.ELE_INFOS(I, 5) nodes of the element
                        mesh.NODES_INFOS[I,:] = np.array(tline.split()[nbinfo:nbinfo+5], dtype=int)
                    if (fileformat == 2): 
                        ele_type = mesh.ELE_INFOS[I,1]
                        num_nodes_per_element = NODES_PER_TYPE_OF_ELEMENT[ele_type-1]
                        NODES_ELEM = np.array(tline.split()[-num_nodes_per_element:], dtype=int)
                        # decrease nodes-number to get 0-indexing
                        NODES_ELEM = NODES_ELEM-1

                    if(mesh.ELE_INFOS[I, 1] == 15): # point
                        mesh.nbPoints = mesh.nbPoints + 1
                        mesh.POINTS[mesh.nbPoints, 0] = IDS[NODES_ELEM[0]]
                        mesh.POINTS[mesh.nbPoints, 1] = mesh.ELE_INFOS[I, tags] 
                    if(mesh.ELE_INFOS[I, 1] == 1): # line
                        mesh.LINES[mesh.nbLines, 0] = IDS[NODES_ELEM[0]]
                        mesh.LINES[mesh.nbLines, 1] = IDS[NODES_ELEM[1]]
                        mesh.LINES[mesh.nbLines, 2] = mesh.ELE_INFOS[I, tags] 
                        mesh.nbLines = mesh.nbLines + 1
                    if(mesh.ELE_INFOS[I, 1] == 2): #triangle
                        mesh.TRIANGLES[mesh.nbTriangles, 0] = IDS[NODES_ELEM[0]]
                        mesh.TRIANGLES[mesh.nbTriangles, 1] = IDS[NODES_ELEM[1]]
                        mesh.TRIANGLES[mesh.nbTriangles, 2] = IDS[NODES_ELEM[2]]
                        mesh.TRIANGLES[mesh.nbTriangles, 3] = mesh.ELE_INFOS[I, tags] 
                        mesh.nbTriangles = mesh.nbTriangles + 1
                    if(mesh.ELE_INFOS[I, 1] == 3): #quadrangle
                        mesh.QUADS[mesh.nbQuads, 0] = IDS[NODES_ELEM[0]]
                        mesh.QUADS[mesh.nbQuads, 1] = IDS[NODES_ELEM[1]]
                        mesh.QUADS[mesh.nbQuads, 2] = IDS[NODES_ELEM[2]]
                        mesh.QUADS[mesh.nbQuads, 3] = IDS[NODES_ELEM[3]]
                        mesh.QUADS[mesh.nbQuads, 4] = mesh.ELE_INFOS[I, tags] 
                        mesh.nbQuads = mesh.nbQuads + 1
                    if(mesh.ELE_INFOS[I, 1] == 4): # tetrahedron
                        mesh.TETS[mesh.nbTets, 0] = IDS[NODES_ELEM[0]]
                        mesh.TETS[mesh.nbTets, 1] = IDS[NODES_ELEM[1]]
                        mesh.TETS[mesh.nbTets, 2] = IDS[NODES_ELEM[2]]
                        mesh.TETS[mesh.nbTets, 3] = IDS[NODES_ELEM[3]]
                        mesh.TETS[mesh.nbTets, 4] = mesh.ELE_INFOS[I, tags] 
                        mesh.nbTets = mesh.nbTets + 1
                    if(mesh.ELE_INFOS[I, 1] == 5): # hexahedron
                        mesh.HEXAS[mesh.nbHexas, 0] = IDS[NODES_ELEM[0]]
                        mesh.HEXAS[mesh.nbHexas, 1] = IDS[NODES_ELEM[1]]
                        mesh.HEXAS[mesh.nbHexas, 2] = IDS[NODES_ELEM[2]]
                        mesh.HEXAS[mesh.nbHexas, 3] = IDS[NODES_ELEM[3]]
                        mesh.HEXAS[mesh.nbHexas, 4] = IDS[NODES_ELEM[4]]
                        mesh.HEXAS[mesh.nbHexas, 5] = IDS[NODES_ELEM[5]]
                        mesh.HEXAS[mesh.nbHexas, 6] = IDS[NODES_ELEM[7]]
                        mesh.HEXAS[mesh.nbHexas, 8] = IDS[NODES_ELEM[8]]
                        mesh.HEXAS[mesh.nbHexas, 9] = mesh.ELE_INFOS[I, tags]
                        mesh.nbHexas = mesh.nbHexas + 1
                    if(mesh.ELE_INFOS[I, 1] == 6): # prism
                        mesh.PRISMS[mesh.nbPrisms, 0] = IDS[NODES_ELEM[0]]
                        mesh.PRISMS[mesh.nbPrisms, 1] = IDS[NODES_ELEM[1]]
                        mesh.PRISMS[mesh.nbPrisms, 2] = IDS[NODES_ELEM[2]]
                        mesh.PRISMS[mesh.nbPrisms, 3] = IDS[NODES_ELEM[3]]
                        mesh.PRISMS[mesh.nbPrisms, 4] = IDS[NODES_ELEM[4]]
                        mesh.PRISMS[mesh.nbPrisms, 5] = IDS[NODES_ELEM[5]]
                        mesh.PRISMS[mesh.nbPrisms, 6] = mesh.ELE_INFOS[I, tags]
                        mesh.nbPrisms = mesh.nbPrisms + 1
                    if(mesh.ELE_INFOS[I, 1] == 7): # pyramid
                        mesh.PYRAMIDS[mesh.nbPyramids, 0] = IDS[NODES_ELEM[0]]
                        mesh.PYRAMIDS[mesh.nbPyramids, 1] = IDS[NODES_ELEM[1]]
                        mesh.PYRAMIDS[mesh.nbPyramids, 2] = IDS[NODES_ELEM[2]]
                        mesh.PYRAMIDS[mesh.nbPyramids, 3] = IDS[NODES_ELEM[3]]
                        mesh.PYRAMIDS[mesh.nbPyramids, 4] = IDS[NODES_ELEM(4)]
                        mesh.PYRAMIDS[mesh.nbPyramids, 5] = IDS[NODES_ELEM[5]]
                        mesh.PYRAMIDS[mesh.nbPyramids, 6] = mesh.ELE_INFOS[I, tags]
                        mesh.nbPyramids = mesh.nbPyramids + 1
                    if(mesh.ELE_INFOS[I, 1] == 8): # second order line
                        mesh.LINES3[mesh.nbLines3, 0] = IDS[NODES_ELEM[0]];
                        mesh.LINES3[mesh.nbLines3, 1] = IDS[NODES_ELEM[1]];
                        mesh.LINES3[mesh.nbLines3, 2] = IDS[NODES_ELEM[2]];
                        mesh.LINES3[mesh.nbLines3, 3] = mesh.ELE_INFOS[I, tags];
                        mesh.nbLines3 = mesh.nbLines3 + 1;
                    if(mesh.ELE_INFOS[I, 1] == 9): # second order triangle
                        mesh.TRIANGLES6[mesh.nbTriangles6, 0] = IDS[NODES_ELEM[0]];
                        mesh.TRIANGLES6[mesh.nbTriangles6, 1] = IDS[NODES_ELEM[1]];
                        mesh.TRIANGLES6[mesh.nbTriangles6, 2] = IDS[NODES_ELEM[2]];
                        mesh.TRIANGLES6[mesh.nbTriangles6, 3] = IDS[NODES_ELEM[3]];
                        mesh.TRIANGLES6[mesh.nbTriangles6, 4] = IDS[NODES_ELEM[4]];
                        mesh.TRIANGLES6[mesh.nbTriangles6, 5] = IDS[NODES_ELEM[5]];
                        mesh.TRIANGLES6[mesh.nbTriangles6, 6] = mesh.ELE_INFOS[I, tags];
                        mesh.nbTriangles6 = mesh.nbTriangles6 + 1;
                tline = fid.readline().rstrip()
                print('Mesh file: elements have been read')
        
        fid.close()
        return mesh
        
