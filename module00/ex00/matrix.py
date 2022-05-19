from copy import deepcopy

class Matrix(object):
    """
    Matrix class to perform both matrix-matrix and matrix-vector operations.
    """
    def __init__(self, user_input):
        """
        Matrix can be initialized with a list of lists or the shape of the matrix.
        """
        if type(user_input) == list and len(user_input) > 0 and type(user_input[0]) == list:
            col_len = len(user_input[0])
            for item in user_input:
                if len(item) != col_len:
                    raise Exception("Columns must be of the same length.")
            self.data = deepcopy(user_input)
            self.shape = (len(user_input), col_len)
        elif type(user_input) == tuple:
            if user_input[0] <= 0 or user_input[1] <= 0:
                raise Exception("Matrix shape must be positive.")
            self.shape = user_input
            self.data = [[0 for x in range(user_input[1])] for y in range(user_input[0])]
        else:
            raise Exception("Matrix can only be initialized with a list of lists or a shape.")
        return

    def __str__(self):
        """
        Prints the data of the matrix.
        """
        txt = "Matrix(" + str(self.data) + ")"
        return txt

    def __repr__(self):
        """
        Represents Matrix object.
        """
        txt = "Matrix(" + str(self.data) + ")"
        return txt

    def __add__(self, matrix):
        """
        Returns the addition of two matrices.
        """
        if type(matrix) != Matrix and matrix.shape != self.shape:
            raise Exception("Adds only matrices of same dimensions.")
        output = list()
        for y in range(self.shape[0]):
            columns = deepcopy(self.data[y])
            for x in range(self.shape[1]):
                columns[x] += matrix.data[y][x]
            output.append(columns)
        return Matrix(output)

    def __radd__(self, matrix):
        """
        Returns the addition of two matrices.
        """
        return self + matrix

    def __sub__(self, matrix):
        """
        Returns the subtraction of two matrices.
        """
        if type(matrix) != Matrix and matrix.shape != self.shape:
            raise Exception("Subtracts only matrices of same dimensions.")
        output = list()
        for y in range(self.shape[0]):
            columns = deepcopy(self.data[y])
            for x in range(self.shape[1]):
                columns[x] -= matrix.data[y][x]
            output.append(columns)
        return Matrix(output)

    def __rsub__(self, matrix):
        """
        Returns the subtraction of two matrices.
        """
        if type(matrix) != Matrix and matrix.shape != self.shape:
            raise Exception("Subtracts only matrices of same dimensions.")
        output = list()
        for y in range(self.shape[0]):
            columns = deepcopy(self.data[y])
            for x in range(self.shape[1]):
                columns[x] = matrix.data[y][x] - columns[x]
            output.append(columns)
        return Matrix(output)

    def __truediv__(self, scalar):
        """
        Returns the division of a Matrix by a scalar.
        """
        if not type(scalar) == int and not type(scalar) == float:
            raise Exception("Divides only matrix by a scalar.")
        output = list()
        for y in range(self.shape[0]):
            columns = deepcopy(self.data[y])
            for x in range(self.shape[1]):
                columns[x] /= scalar
            output.append(columns)
        return Matrix(output)

    def __rtruediv__(self, scalar):
        """
        Returns the division of a Matrix by a scalar.
        """
        if not type(scalar) == int and not type(scalar) == float:
            raise Exception("Divides only matrix by a scalar.")
        output = list()
        for y in range(self.shape[0]):
            columns = deepcopy(self.data[y])
            for x in range(self.shape[1]):
                columns[x] = scalar / columns[x]
            output.append(columns)
        return Matrix(output)

    def __mul__(self, value):
        """
        Multiplies scalars, vectors and matrices.
        """
        output = list()
        if type(value) == int or type(value) == float:
            for y in range(self.shape[0]):
                columns = deepcopy(self.data[y])
                for x in range(self.shape[1]):
                    columns[x] *= value
                output.append(columns)
        elif type(value) == Matrix or type(value) == Vector:
            if self.shape[1] != value.shape[0]:
                raise Exception("Matrices dimensions are incompatible for multiplication.")
            for y in range(self.shape[0]):
                columns = list()
                for x in range(value.shape[1]):
                    result = 0
                    for t in range(self.shape[1]):
                        result += (self.data[y][t] * value.data[t][x])
                    columns.append(result)
                output.append(columns)
            if type(value) == Vector:
                return Vector(output)
        else:
            raise Exception("Matrix can only be multiplied by scalars, vectors and matrices.")
        return Matrix(output)

    def __rmul__(self, value):
        """
        Multiplies scalars, vectors and matrices.
        """
        return self * value

    def T(self):
        """
        Returns the transpose of the Matrix object.
        """
        output = list()
        for x in range(self.shape[1]):
            columns = list()
            for y in range(self.shape[0]):
                columns.append(self.data[y][x])
            output.append(columns)
        return Matrix(output)

class Vector(Matrix):
    """
    Vector class that inherits from Matrix class.
    """
    def __init__(self, user_input):
        """
        Vector class must have a single row or a single column.
        """
        if type(user_input) == list and len(user_input) == 0:
            raise Exception("Vector must have dimension (1,n) or (n,1).")
        if type(user_input) == list and len(user_input) != 1 and len(user_input[0]) != 1:
            raise Exception("Vector must have dimension (1,n) or (n,1).")
        if type(user_input) == tuple and user_input[0] != 1 and user_input[1] != 1:
            raise Exception("Vector must have dimension (1,n) or (n,1).")
        super().__init__(user_input)
        return

    def __str__(self):
        """
        Prints the data of the vector.
        """
        txt = "Vector(" + str(self.data) + ")"
        return txt

    def __add__(self, vector):
        """
        Returns the addition of two vectors.
        """
        return Vector((Matrix(self.data) + Matrix(vector.data)).data)

    def __radd__(self, vector):
        """
        Returns the addition of two vectors.
        """
        return self + vector

    def __sub__(self, vector):
        """
        Returns the subtraction of two vectors.
        """
        return Vector((Matrix(self.data) - Matrix(vector.data)).data)

    def __rsub__(self, vector):
        """
        Returns the subtraction of two vectors.
        """
        return Vector((Matrix(vector.data) - Matrix(self.data)).data)

    def __truediv__(self, scalar):
        """
        Returns the division of a Vector by a scalar.
        """
        return Vector((Matrix(self.data) / scalar).data)

    def __rtruediv__(self, scalar):
        """
        Returns the division of a Vector by a scalar.
        """
        return Vector((scalar / Matrix(self.data)).data)

    def __mul__(self, value):
        """
        Returns a matrix with the multiplication of vectors and scalars.
        """
        return Matrix(self.data) * value 

    def __rmul__(self, value):
        """
        Returns a matrix with the multiplication of vectors and scalars.
        """
        return Matrix(self.data) * value

    def dot(self, v):
        """
        Returns the dot product between two vectors.
        """
        if not type(v) == Vector:
            raise Exception("Vector can only dot multiply with another vector.")
        if self.shape != v.shape:
            raise Exception("Vectors must be of the same shape.")
        output = deepcopy(self.data)
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                output[y][x] = self.data[y][x] * v.data[y][x]
        return output
