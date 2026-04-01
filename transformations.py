import numpy as np

def eul2dcm(angles, sequence, degrees):
    """ Converts a set of euler angles into a direction cosine matrix

    Args:
        angles (tuple) - euler angles (theta1, theta2, theta3) in radians or degrees
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
        angle = angles(index)
        dcm = get_rotation_matrix(angle, char) @ dcm 

    return dcm
    
def get_rotation_matrix(angle, axis):
    """ Converts an angle into a rotation matrix

    Args:
        angle (float) - rotation angle [rad]
        axis (char) - rotation axis
    Ret:
        rotm (3x3 numpy array) - rotation matrix
    """
    s = np.sin(angle)
    c = np.cos(angle)

    if axis=='1':
        rotm = np.array([1,0,0], [0, c, s], [0, -s, c])
    if axis=='2':
        rotm = np.array([c, 0, -s], [0,1,0], [s, 0, c])
    if axis=='3':
        rotm = np.array([c, s, 0], [-s, c, 0], [0,0,1])

    return rotm