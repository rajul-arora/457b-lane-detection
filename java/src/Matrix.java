import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;

class Matrix<T> implements Iterable {

    private int width;
    private int height;
    private ArrayList<ArrayList<T>> data;

    public Matrix(int[] dim) {
        this(dim[0], dim[1]);
    }

    public Matrix(int width, int height) {
        this.width = width;
        this.height = height;
        this.data = new ArrayList();

        for(int i = 0; i < height; i++) {
            this.data.add(new ArrayList<T>());
        }
    }

    public int[] size() {
        return new int[]{width, height};
    }

    public int width() {
        return this.width;
    }

    public int height() {
        return height;
    }

    public void set(int i, int j, T data) {
        this.data.get(i).set(j, data);
    }

    public T get(int i, int j) {
        return this.data.get(i).get(j);
    }

    public static <T> Matrix convert(T[][] jaggedArray) {

        int height = jaggedArray.length;
        int width = jaggedArray[0].length;
        Matrix<T> matrix = new Matrix(width, height);

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                matrix.set(i, j, jaggedArray[i][j]);
            }
        }

        return matrix;
    }

    public void flatten(T[] array) {

        ArrayList<T> vector = new ArrayList<T>();
        for (int y = 0; y < this.height(); y++) {
            for (int x = 0; x < this.width(); x++) {
                vector.add(this.get(y, x));
            }
        }

        vector.toArray(array);
    }

//    public Matrix multiply(double val) {
//
//        Matrix m = new Matrix(this.size());
//        for (int y = 0; y < this.height(); y++) {
//            for (int x = 0; x < this.width(); x++) {
//                m.set(y, x, (int) (this.get(y, x) * val));
//            }
//        }
//
//        return m;
//    }

    public void toJaggedArray(T[][] array) {
        for( int y =0; y < this.height; y++) {
            this.data.get(y).toArray(array[y]);
        }
    }

    @Override
    public Iterator iterator() {
        return this.data.iterator();
    }

    @Override
    public String toString() {

        String size = this.width + ", " + this.height;
        String output = "Matrix <" + size + "> [\n";
        for (ArrayList<T> row : this.data) {
            for (T val : row) {
                output += val + ", ";
            }

            output += "\n";
        }

        output += "]";
        return output;
    }
}