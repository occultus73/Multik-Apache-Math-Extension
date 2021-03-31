/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.commons.math4.linear

import org.apache.commons.math4.util.Precision
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Class transforming a general real matrix to Hessenberg form.
 *
 * A m  m matrix A can be written as the product of three matrices: A = P
 *  H  P<sup>T</sup> with P an orthogonal matrix and H a Hessenberg
 * matrix. Both P and H are m  m matrices.
 *
 * Transformation to Hessenberg form is often not a goal by itself, but it is an
 * intermediate step in more general decomposition algorithms like
 * [eigen decomposition][EigenDecomposition]. This class is therefore
 * intended for internal use by the library and is not public. As a consequence
 * of this explicitly limited scope, many methods directly returns references to
 * internal arrays, not copies.
 *
 * This class is based on the method orthes in class EigenvalueDecomposition
 * from the [JAMA](http://math.nist.gov/javanumerics/jama/) library.
 *
 * @see [MathWorld](http://mathworld.wolfram.com/HessenbergDecomposition.html)
 *
 * @see [Householder Transformations](http://en.wikipedia.org/wiki/Householder_transformation)
 *
 * @since 3.1
 */
internal class HessenbergTransformer(matrix: RealMatrix) {
    /** Householder vectors.  */
    private val householderVectors: Array<DoubleArray>

    /** Temporary storage vector.  */
    private val ort: DoubleArray

    /** Cached value of P.  */
    private var cachedP: NDArray<Double, D2>?

    /** Cached value of Pt.  */
    private var cachedPt: NDArray<Double, D2>?

    /** Cached value of H.  */
    private var cachedH: NDArray<Double, D2>?

    /**
     * Returns the matrix P of the transform.
     *
     * P is an orthogonal matrix, i.e. its inverse is also its transpose.
     *
     * @return the P matrix
     */
    fun getP(): NDArray<Double, D2> {
        if (cachedP == null) {
            val n = householderVectors.size
            val high = n - 1
            val pa = Array(n) { DoubleArray(n) }
            for (i in 0 until n) {
                for (j in 0 until n) {
                    pa[i][j] = if (i == j) 1.0 else 0.0
                }
            }
            for (m in high - 1 downTo 1) {
                if (householderVectors[m][m - 1] != 0.0) {
                    for (i in m + 1..high) {
                        ort[i] = householderVectors[i][m - 1]
                    }
                    for (j in m..high) {
                        var g = 0.0
                        for (i in m..high) {
                            g += ort[i] * pa[i][j]
                        }

                        // Double division avoids possible underflow
                        g = g / ort[m] / householderVectors[m][m - 1]
                        for (i in m..high) {
                            pa[i][j] += g * ort[i]
                        }
                    }
                }
            }
            cachedP = pa.toNDArray()
        }
        return cachedP!!
    }

    /**
     * Returns the transpose of the matrix P of the transform.
     *
     * P is an orthogonal matrix, i.e. its inverse is also its transpose.
     *
     * @return the transpose of the P matrix
     */
    fun getPT(): NDArray<Double, D2> {
        if (cachedPt == null) {
            cachedPt = getP().transpose()
        }

        // return the cached matrix
        return cachedPt!!
    }

    /**
     * Returns the Hessenberg matrix H of the transform.
     *
     * @return the H matrix
     */
    fun getH(): NDArray<Double, D2> {
        if (cachedH == null) {
            val m = householderVectors.size
            val h = Array(m) { DoubleArray(m) }
            for (i in 0 until m) {
                if (i > 0) {
                    // copy the entry of the lower sub-diagonal
                    h[i][i - 1] = householderVectors[i][i - 1]
                }

                // copy upper triangular part of the matrix
                for (j in i until m) {
                    h[i][j] = householderVectors[i][j]
                }
            }
            cachedH = h.toNDArray()
        }

        // return the cached matrix
        return cachedH!!
    }

    /**
     * Get the Householder vectors of the transform.
     *
     * Note that since this class is only intended for internal use, it returns
     * directly a reference to its internal arrays, not a copy.
     *
     * @return the main diagonal elements of the B matrix
     */
    fun getHouseholderVectorsRef(): Array<DoubleArray> {
        return householderVectors
    }

    /**
     * Transform original matrix to Hessenberg form.
     *
     * Transformation is done using Householder transforms.
     */
    private fun transform() {
        val n = householderVectors.size
        val high = n - 1
        for (m in 1 until high) {
            // Scale column.
            var scale = 0.0
            for (i in m..high) {
                scale += abs(householderVectors[i][m - 1])
            }
            if (!Precision.equals(scale, 0.0)) {
                // Compute Householder transformation.
                var h = 0.0
                for (i in high downTo m) {
                    ort[i] = householderVectors[i][m - 1] / scale
                    h += ort[i] * ort[i]
                }
                val g: Double = if (ort[m] > 0) -sqrt(h) else sqrt(h)
                h -= ort[m] * g
                ort[m] -= g

                // Apply Householder similarity transformation
                // H = (I - u*u' / h) * H * (I - u*u' / h)
                for (j in m until n) {
                    var f = 0.0
                    for (i in high downTo m) {
                        f += ort[i] * householderVectors[i][j]
                    }
                    f /= h
                    for (i in m..high) {
                        householderVectors[i][j] -= f * ort[i]
                    }
                }
                for (i in 0..high) {
                    var f = 0.0
                    for (j in high downTo m) {
                        f += ort[j] * householderVectors[i][j]
                    }
                    f /= h
                    for (j in m..high) {
                        householderVectors[i][j] -= f * ort[j]
                    }
                }
                ort[m] = scale * ort[m]
                householderVectors[m][m - 1] = scale * g
            }
        }
    }

    /**
     * Build the transformation to Hessenberg form of a general matrix.
     *
     * @param matrix matrix to transform
     *
     */
    init {
        if (!matrix.isSquare()) {
            throw ArithmeticException("Non Square Matrix ${matrix.shape}")
        }
        val m = matrix.getRowDimension()
        householderVectors = matrix.getData()
        ort = DoubleArray(m)
        cachedP = null
        cachedPt = null
        cachedH = null

        // transform matrix
        transform()
    }
}