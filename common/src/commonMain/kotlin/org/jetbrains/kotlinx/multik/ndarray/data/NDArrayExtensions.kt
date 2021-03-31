package org.jetbrains.kotlinx.multik.ndarray.data

import org.jetbrains.kotlinx.multik.api.empty
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import kotlin.jvm.JvmName
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

// For Apache Commons Math translation.
typealias RealMatrix = MultiArray<Double, D2>
typealias RealVector = MultiArray<Double, D1>

// Determines if Matrix is a square.
fun RealMatrix.isSquare(): Boolean {
    val nRows = shape[0]
    val nCols = shape[1]
    return nRows == nCols
}

// Returns number of rows in a Matrix
fun RealMatrix.getRowDimension() = shape[0]

// Returns number of columns in Matrix
fun RealMatrix.getColumnDimension() = shape[1]

// Return Matrix as an array of double arrays (useful for manual modification).
fun RealMatrix.getData(): Array<DoubleArray> {
    return List(shape[0]) { this[it].toList() }.run {
        Array(size) { this[it].toDoubleArray() }
    }
}

// Return Vector as a double array.
fun RealVector.getData() = data.getDoubleArray()

// Convert an array of double arrays into an ND array (Real Matrix).
fun Array<DoubleArray>.toNDArray(): NDArray<Double, D2> {
    return mk.ndarray(List(size) { this[it].toList().map { d -> if (d == -0.0) 0.0 else d } })
}

// Convert a double array into an ND array (Real Vector)
fun DoubleArray.toNDArray() = mk.ndarray(this)

// Extract a certain area of a matrix to make another matrix.
fun RealMatrix.getSubMatrix(
    startRow: Int,
    endRow: Int,
    startColumn: Int,
    endColumn: Int
): RealMatrix {
    return mk.ndarray(mutableListOf<List<Double>>().also { subMatrix ->
        for (rowIndex in startRow..endRow) {
            subMatrix.add(mutableListOf<Double>().also { row ->
                for (columnIndex in startColumn..endColumn) {
                    row.add(this[rowIndex][columnIndex])
                }
            })
        }
    })
}

// Extract a certain column from a matrix.
fun RealMatrix?.getColumnVector(index: Int): RealVector {
    if (this == null) throw NullPointerException("getColumnVector called on null matrix")
    return getData().let { array ->
        DoubleArray(shape[0]) { array[it][index] }.toNDArray()
    }
}

// Set a certain column in a matrix. Return new matrix.
fun RealMatrix?.setColumnVector(
    index: Int,
    columnVector: RealVector
): RealMatrix {
    if (this == null) throw NullPointerException("setColumnVector called on null matrix")
    if (columnVector.size != shape[0]) throw IllegalArgumentException("Column Vector is wrong size.")
    return getData().let { array ->
        for (i in array.indices) {
            array[i][index] = columnVector[i]
        }
        array.toNDArray()
    }
}

// Set a certain column in a matrix and return new matrix.
fun RealMatrix?.setRowVector(
    index: Int,
    rowVector: RealVector
): RealMatrix {
    if (this == null) throw NullPointerException("setColumnVector called on null matrix")
    if (rowVector.size != shape[1]) throw IllegalArgumentException("Row Vector is wrong size.")
    return getData().let { array ->
        for (i in array.indices) {
            array[index][i] = rowVector[i]
        }
        array.toNDArray()
    }
}

// Create a diagonal matrix from a given double array.
fun mk.createRealDiagonalMatrix(diagonal: DoubleArray): RealMatrix {
    val m = empty<Double, D2>(diagonal.size, diagonal.size).getData()
    for (i in diagonal.indices) {
        m[i][i] = diagonal[i]
    }
    return m.toNDArray()
}

// The Apache "operate" RealMatrix method, rather too difficult to succinctly explain here.
fun RealMatrix.operate(v: RealVector): RealVector {
    val nRows: Int = this.getRowDimension()
    val nCols: Int = this.getColumnDimension()
    if (v.size != nCols) {
        throw IllegalArgumentException("Dimension mismatch: ${v.size}, $nCols")
    }
    val out = DoubleArray(nRows)
    for (row in 0 until nRows) {
        val dataRow: DoubleArray = getData()[row]
        var sum = 0.0
        for (i in 0 until nCols) {
            sum += dataRow[i] * v[i]
        }
        out[row] = sum
    }
    return out.toNDArray()
}

// Returns the norm value of a matrix.
@JvmName("getNormRealMatrix")
fun RealMatrix.getNorm(): Double {
    /** Sum of absolute values on one column.  */
    var columnSum = 0.0

    /** Maximal sum across all columns.  */
    var maxColSum = 0.0
    val rows = getRowDimension()
    val columns = getColumnDimension()
    val endRow = (rows - 1).toDouble()
    for (column in 0 until columns) {
        for (row in 0 until rows) {
            columnSum += abs(this[row][column])
            if (row.toDouble() == endRow) {
                maxColSum = max(maxColSum, columnSum)
                columnSum = 0.0
            }
        }
    }
    return maxColSum
}

// Returns the norm value of a vector.
@JvmName("getNormRealVector")
fun RealVector.getNorm(): Double {
    var sum = 0.0
    val it = iterator()
    while (it.hasNext()) {
        val e = it.next()
        sum += e * e
    }
    return sqrt(sum)
}