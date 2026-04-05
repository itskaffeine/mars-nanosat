import numpy as np
def create_submission_txt(filename:str, value, precision=6) -> None:
    """
    Create a submission text based on a filename  and a value

    If the value is a list or matrix, it will flatten it properly to be printed
    """
    np.set_printoptions(suppress=True, precision=precision)
    with open(filename, "w") as f:
        if type(value) == float or type(value) == int: 
            print(value, file=f)
        elif type(value) == list:
            print(*value, file=f)
        elif type(value) == np.ndarray:  
            print(*value.flatten(), file=f)