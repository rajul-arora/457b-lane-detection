class Matrix:
    def __init__(self, dim = [3, 3]):
        self.dim = dim
        self.data = [[0 for x in range(dim[0])] for y in range(dim[1])]

    def size(self):
        return self.dim

    def width(self):
        return self.dim[0]

    def height(self):
        return self.dim[1]

    @staticmethod
    def convert(jaggedArray):
        
        height = len(jaggedArray)
        width = len(jaggedArray[0])
        matrix = Matrix([width, height])

        for i in range(height):
            for j in range(width):
                matrix.data[i][j] = jaggedArray[i][j]

        return matrix

    def toJaggedArray(self):
        return self.data

    def __delitem__(self, key):
        del self.data[key]

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __iter__(self):
        return self.data.__iter__

    def __str__(self):

        size = str(self.dim[0]) + ", " + str(self.dim[1])
        output = "Matrix <" + size + "> [\n"
        for row in self.data:
            for val in row:
                output += str(val) + ", "

            output += "\n"

        output += "]"
        return output
    
    def __repr__(self):
        return self.__str__()