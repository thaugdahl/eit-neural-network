def multiply_variables(row):
    """
    Returns the multiplication of the first "num_vars" elements from the row.
    :param row: The row with values
    :return: The multiplication of the variables
    """
    final_value = row[-1][0]
    for i in range(1, len(row[-1]) - 1):
        final_value *= row[-1][i]
    return final_value

def xy(vect):
    return vect[:,:,0]*vect[:,:,1]

def inject_constant(row):
    """
    Silly function that returns 1.
    :param row: Not meaningful
    :return: Returns the int 1
    """
    return 1

    