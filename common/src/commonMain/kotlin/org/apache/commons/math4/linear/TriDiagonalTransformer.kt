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

import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.math.sqrt

/**
 * Class transforming a symmetrical matrix to tridiagonal shape.
 *
 * A symmetrical m  m matrix A can be written as the product of three matrices:
 * A = Q  T  Q<sup>T</sup> with Q an orthogonal matrix and T a symmetrical
 * tridiagonal matrix. Both Q and T are m  m matrices.
 *
 * This implementation only uses the upper part of the matrix, the part below the
 * diagonal is not accessed at all.
 *
 * Transformation to tridiagonal shape is often not a goal by itself, but it is
 * an intermediate step in more general decomposition algorithms like [ ]. This class is therefore intended for internal
 * use by the library and is not public. As a consequence of this explicitly limited scope,
 * many methods directly returns references to internal arrays, not copies.
 * @since 2.0
 */

/**
 * Build the transformation to tridiagonal shape of a symmetrical matrix.
 *
 * The specified matrix is assumed to be symmetrical without any check.
 * Only the upper triangular part of the matrix is used.
 *
 * @param matrix Symmetrical matrix to transform.
 *
 */
internal class TriDiagonalTransformer(matrix: RealMatrix) {
    /** Householder vectors.  */
    private val householderVectors: Array<DoubleArray>

    /** Main diagonal.  */
    private val main: DoubleArray

    /** Secondary diagonal.  */
    private val secondary: DoubleArray

    /** Cached value of Q.  */
    private var cachedQ: RealMatrix?

    /** Cached value of Qt.  */
    private var cachedQt: NDArray<Double, D2>?

    /** Cached value of T.  */
    private var cachedT: NDArray<Double, D2>?

    /**
     * Returns the matrix Q of the transform.
     *
     * Q is an orthogonal matrix, i.e. its transpose is also its inverse.
     * @return the Q matrix
     */
    fun getQ(): RealMatrix {
        if (cachedQ == null) {
            cachedQ = getQT().transpose()
        }
        return cachedQ!!
    }

    /**
     * Returns the transpose of the matrix Q of the transform.
     *
     * Q is an orthogonal matrix, i.e. its transpose is also its inverse.
     * @return the Q matrix
     */
    fun getQT(): RealMatrix {
        if (cachedQt == null) {
            val m = householderVectors.size
            val qta = Array(m) { DoubleArray(m) }

            // build up first part of the matrix by applying Householder transforms
            for (k in m - 1 downTo 1) {
                val hK = householderVectors[k - 1]
                qta[k][k] = 1.0
                if (hK[k] != 0.0) {
                    val inv: Double = 1.0 / (secondary[k - 1] * hK[k])
                    var beta: Double = 1.0 / secondary[k - 1]
                    qta[k][k] = 1.0 + beta * hK[k]
                    for (i in k + 1 until m) {
                        qta[k][i] = beta * hK[i]
                    }
                    for (j in k + 1 until m) {
                        beta = 0.0
                        for (i in k + 1 until m) {
                            beta += qta[j][i] * hK[i]
                        }
                        beta *= inv
                        qta[j][k] = beta * hK[k]
                        for (i in k + 1 until m) {
                            qta[j][i] += beta * hK[i]
                        }
                    }
                }
            }
            qta[0][0] = 1.0
            cachedQt = qta.toNDArray()
        }

        // return the cached matrix
        return cachedQt!!
    }

    /**
     * Returns the tridiagonal matrix T of the transform.
     * @return the T matrix
     */
    fun getT(): RealMatrix {
        if (cachedT == null) {
            val m: Int = main.size
            val ta = Array(m) { DoubleArray(m) }
            for (i in 0 until m) {
                ta[i][i] = main[i]
                if (i > 0) {
                    ta[i][i - 1] = secondary[i - 1]
                }
                if (i < main.size - 1) {
                    ta[i][i + 1] = secondary[i]
                }
            }
            cachedT = ta.toNDArray()
        }

        // return the cached matrix
        return cachedT!!
    }

    /**
     * Get the Householder vectors of the transform.
     *
     * Note that since this class is only intended for internal use,
     * it returns directly a reference to its internal arrays, not a copy.
     * @return the main diagonal elements of the B matrix
     */
    fun getHouseholderVectorsRef(): Array<DoubleArray> {
        return householderVectors
    }

    /**
     * Get the main diagonal elements of the matrix T of the transform.
     *
     * Note that since this class is only intended for internal use,
     * it returns directly a reference to its internal arrays, not a copy.
     * @return the main diagonal elements of the T matrix
     */
    fun getMainDiagonalRef(): DoubleArray {
        return main
    }

    /**
     * Get the secondary diagonal elements of the matrix T of the transform.
     *
     * Note that since this class is only intended for internal use,
     * it returns directly a reference to its internal arrays, not a copy.
     * @return the secondary diagonal elements of the T matrix
     */
    fun getSecondaryDiagonalRef(): DoubleArray {
        return secondary
    }

    /**
     * Transform original matrix to tridiagonal form.
     *
     * Transformation is done using Householder transforms.
     */
    private fun transform() {
        val m = householderVectors.size
        val z = DoubleArray(m)
        for (k in 0 until m - 1) {

            //zero-out a row and a column simultaneously
            val hK = householderVectors[k]
            main[k] = hK[k]
            var xNormSqr = 0.0
            for (j in k + 1 until m) {
                val c: Double = hK[j]
                xNormSqr += c * c
            }
            val a = if (hK[k + 1] > 0) -sqrt(xNormSqr) else sqrt(xNormSqr)
            secondary[k] = a
            if (a != 0.0) {
                // apply Householder transform from left and right simultaneously
                hK[k + 1] -= a
                val beta: Double = -1 / (a * hK[k + 1])

                // compute a = beta A v, where v is the Householder vector
                // this loop is written in such a way
                //   1) only the upper triangular part of the matrix is accessed
                //   2) access is cache-friendly for a matrix stored in rows
                z.fill(0.0, k + 1, m)
                for (i in k + 1 until m) {
                    val hI = householderVectors[i]
                    val hKI: Double = hK[i]
                    var zI: Double = hI[i] * hKI
                    for (j in i + 1 until m) {
                        val hIJ: Double = hI[j]
                        zI += hIJ * hK[j]
                        z[j] += hIJ * hKI
                    }
                    z[i] = beta * (z[i] + zI)
                }

                // compute gamma = beta vT z / 2
                var gamma = 0.0
                for (i in k + 1 until m) {
                    gamma += z[i] * hK[i]
                }
                gamma *= beta / 2

                // compute z = z - gamma v
                for (i in k + 1 until m) {
                    z[i] -= gamma * hK[i]
                }

                // update matrix: A = A - v zT - z vT
                // only the upper triangular part of the matrix is updated
                for (i in k + 1 until m) {
                    val hI = householderVectors[i]
                    for (j in i until m) {
                        hI[j] -= hK[i] * z[j] + z[i] * hK[j]
                    }
                }
            }
        }
        main[m - 1] = householderVectors[m - 1][m - 1]
    }

    init {
        if (!matrix.isSquare()) {
            throw ArithmeticException("Non Square Matrix ${matrix.shape}")
        }
        val m = matrix.getRowDimension()
        householderVectors = matrix.getData()
        main = DoubleArray(m)
        secondary = DoubleArray(m - 1)
        cachedQ = null
        cachedQt = null
        cachedT = null

        // transform matrix
        transform()
    }
}