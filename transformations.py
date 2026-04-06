import numpy as np
import math

def eul2dcm(angles, sequence, degrees):
    """ Converts a set of euler angles into a direction cosine matrix

    Params:
        angles (3x1 tuple) - euler angles (theta1, theta2, theta3) in radians or degrees
        sequence (string) - angle rotation sequence (i.e. "313", "321")
        degrees (bool) - units (==True if using degrees)

    Ret:
        dcm (3x3 numpy array) - direction consine matrix
    """

    dcm = np.eye(3)

    if degrees:
        angles = np.radians(angles) # put everything in radians for ease

    # iteratively find the rotation matrix and premultiply
    for index, char in enumerate(sequence):
        angle = angles[index]
        dcm = get_rotation_matrix(angle, char) @ dcm 

    return dcm
    
def get_rotation_matrix(angle, axis):
    """ Computes a rotation matrix for a rotation about a given axis

    Params:
        angle (float) - rotation angle [rad]
        axis (char) - rotation axis
    Ret:
        rotm (3x3 numpy array) - rotation matrix
    """
    s = np.sin(angle)
    c = np.cos(angle)

    if axis=='1':
        rotm = np.array([[1,0,0], [0, c, s], [0, -s, c]])
    if axis=='2':
        rotm = np.array([[c, 0, -s], [0,1,0], [s, 0, c]])
    if axis=='3':
        rotm = np.array([[c, s, 0], [-s, c, 0], [0,0,1]])

    return rotm

def axes2dcm(v1, v2, sequence):
    """ Computes a direction cosine matrix describing the rotation of a frame given two basis vectors.

    Params:
        v1 (3x1 numpy array) - vector of the rotated frame
        v2 (3x1 numpy array) - vector of the rotated frame orthogonal to v1
        sequence (string) - ordered tag assigning axes to the basis vectors "[v1][v2]"
    Ret:
        dcm (3x3 numpy array) - direction consine matrix
    """
    v1 = v1/np.linalg.norm(v1)
    v2 = v2/np.linalg.norm(v2)

    if sequence == "xy": dcm = np.array([v1, v2, np.cross(v1,v2)])
    if sequence == "yz": dcm = np.array([np.cross(v1,v2), v1, v2])
    if sequence == "zx": dcm = np.array([v2, np.cross(v1,v2), v1])

    if sequence == "yx": dcm = np.array([v2, v1, np.cross(v2,v1)])
    if sequence == "zy": dcm = np.array([np.cross(v2,v1), v2, v1])
    if sequence == "xz": dcm = np.array([v1, np.cross(v2,v1), v2])

    return dcm

def C2EP(C):
    """
    C2EP

    	Q = C2EP(C) translates the 3x3 direction cosine matrix
    	C into the corresponding 4x1 Euler parameter vector Q,
    	where the first component of Q is the non-dimensional
    	Euler parameter Beta_0 >= 0. Transformation is done
        using the Stanley method.

    :Authors:
        Hanspeter Schaub
        John L. Junkins
    """

    tr = np.trace(C)
    b2 = np.matrix('1.;0.;0.;0.')
    b2[0,0] = (1+tr)/4
    b2[1,0] = (1+2*C[0,0]-tr)/4
    b2[2,0] = (1+2*C[1,1]-tr)/4
    b2[3,0] = (1+2*C[2,2]-tr)/4

    case = np.argmax(b2)
    b = b2
    if   case == 0:
            b[0,0] = math.sqrt(b2[0,0])
            b[1,0] = (C[1,2]-C[2,1])/4/b[0,0]
            b[2,0] = (C[2,0]-C[0,2])/4/b[0,0]
            b[3,0] = (C[0,1]-C[1,0])/4/b[0,0]
    elif case == 1:
            b[1,0] = math.sqrt(b2[1,0])
            b[0,0] = (C[1,2]-C[2,1])/4/b[1,0]
            if b[0,0]<0:
                b[1,0] = -b[1,0]
                b[0,0] = -b[0,0]
            b[2,0] = (C[0,1]+C[1,0])/4/b[1,0]
            b[3,0] = (C[2,0]+C[0,2])/4/b[1,0]
    elif case == 2:
            b[2,0] = math.sqrt(b2[2,0])
            b[0,0] = (C[2,0]-C[0,2])/4/b[2,0]
            if b[0,0]<0:
                b[2,0] = -b[2,0];
                b[0,0] = -b[0,0];
            b[1,0] = (C[0,1]+C[1,0])/4/b[2,0]
            b[3,0] = (C[1,2]+C[2,1])/4/b[2,0]
    elif case == 3:
            b[3,0] = math.sqrt(b2[3,0])
            b[0,0] = (C[0,1]-C[1,0])/4/b[3,0]
            if b[0,0]<0:
                b[3,0] = -b[3,0]
                b[0,0] = -b[0,0]
            b[1,0] = (C[2,0]+C[0,2])/4/b[3,0]
            b[2,0] = (C[1,2]+C[2,1])/4/b[3,0]
    return b

def MRP2C(q):
    """
    MRP2C

    	C = MRP2C(Q) returns the direction cosine
    	matrix in terms of the 3x1 MRP vector Q.
         
    :Authors:
        Hanspeter Schaub
        John L. Junkins
    """

    q1 = q[0];
    q2 = q[1];
    q3 = q[2];

    d1 = (q.T@q);
    S = 1-d1;
    d = (1+d1)*(1+d1);
    C = np.matrix("0. 0. 0.;0. 0. 0.;0. 0. 0.");
    C[0,0] = 4*(2*q1*q1-d1)+S*S;
    C[0,1] = 8*q1*q2+4*q3*S;
    C[0,2] = 8*q1*q3-4*q2*S;
    C[1,0] = 8*q2*q1-4*q3*S;
    C[1,1] = 4*(2*q2*q2-d1)+S*S;
    C[1,2] = 8*q2*q3+4*q1*S;
    C[2,0] = 8*q3*q1+4*q2*S;
    C[2,1] = 8*q3*q2-4*q1*S;
    C[2,2] = 4*(2*q3*q3-d1)+S*S;
    C = C/d;

    return C;

def C2MRP(C):
    """
    C2MRP

    	Q = C2MRP(C) translates the 3x3 direction cosine matrix
    	C into the corresponding 3x1 MRP vector Q where the
    	MRP vector is chosen such that |Q| <= 1.
    """

    b = C2EP(C);

    q = np.matrix('0.;0.;0.');
    q[0] = b[1,0]/(1+b[0,0]);
    q[1] = b[2,0]/(1+b[0,0]);
    q[2] = b[3,0]/(1+b[0,0]);

    return q;

def subMRP(q1,q2):
    """
    subMRP(Q1,Q2)

    	Q = subMRP(Q1,Q2) provides the MRP vector
    	which corresponds to relative rotation from Q2
    	to Q1.
         
    Modified by Hugh MacLaughlin

    :Authors:
        Hanspeter Schaub
        John L. Junkins
    """
    q1 = np.asarray(q1).flatten()
    q2 = np.asarray(q2).flatten()

    dot1 = np.dot(q1,q1)
    dot2 = np.dot(q2,q2)
    dot12 = np.dot(q1,q2)

    q = (1-dot2)*q1-(1-dot1)*q2+2*np.cross(q1,q2)
    q = q/(1+ dot1 * dot2 + 2*dot12)

    mag = np.dot(q, q)
    
    if mag > 1.0:
        q = -q / mag

    return q