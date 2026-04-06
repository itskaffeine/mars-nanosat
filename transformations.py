import numpy as np

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