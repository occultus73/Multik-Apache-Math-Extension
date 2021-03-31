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

import org.jetbrains.kotlinx.multik.ndarray.data.RealMatrix
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.getRowDimension
import org.jetbrains.kotlinx.multik.ndarray.data.isSquare
import kotlin.math.abs
import kotlin.math.max

/**
 * A collection of static methods that operate on or return matrices.
 *
 */
object MatrixUtils {

    /**
     * Checks whether a matrix is symmetric, within a given relative tolerance.
     *
     * @param matrix Matrix to check.
     * @param relativeTolerance Tolerance of the symmetry check.
     * the matrix is not symmetric.
     * @return `true` if `matrix` is symmetric.
     */
    fun isSymmetric(matrix: RealMatrix, relativeTolerance: Double): Boolean {
        if (!matrix.isSquare()) throw ArithmeticException("Non Square Matrix ${matrix.shape}")
        val rows = matrix.getRowDimension()
        for (i in 0 until rows) {
            for (j in i + 1 until rows) {
                val mij: Double = matrix[i][j]
                val mji: Double = matrix[j][i]
                if (abs(mij - mji) > max(abs(mij), abs(mji)) * relativeTolerance) return false
            }
        }
        return true
    }

    /**
     * Computes the inverse of the given matrix.
     *
     *
     * By default, the inverse of the matrix is computed using the QR-decomposition,
     * unless a more efficient method can be determined for the input matrix.
     *
     *
     * Note: this method will use a singularity threshold of 0,
     * use [.inverse] if a different threshold is needed.
     *
     * @param matrix Matrix whose inverse shall be computed
     * @return the inverse of `matrix`
     *
     *
     *
     * @since 3.3
     */
    fun inverse(matrix: RealMatrix, threshold: Double = 0.0): RealMatrix {
        if (!matrix.isSquare()) throw ArithmeticException("Non Square Matrix ${matrix.shape}")
        val decomposition = QRDecomposition(matrix, threshold)
        return decomposition.getSolver().getInverse()
    }

}