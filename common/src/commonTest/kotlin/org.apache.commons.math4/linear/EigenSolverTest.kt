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

import org.apache.commons.math4.TestUtils.assertEquals
import org.apache.commons.math4.util.Precision
import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlin.test.fail

class EigenSolverTest {
    private val bigSingular: Array<DoubleArray> = arrayOf(
        doubleArrayOf(1.0, 2.0, 3.0, 4.0),
        doubleArrayOf(2.0, 5.0, 3.0, 4.0),
        doubleArrayOf(7.0, 3.0, 256.0, 1930.0),
        doubleArrayOf(3.0, 7.0, 6.0, 8.0)
    ) // 4th row = 1st + 2nd

    /** test non invertible matrix  */
    @Test
    fun testNonInvertible() {
        val r = Random(9994100315209L)
        val m: RealMatrix =
            EigenDecompositionTest.createTestMatrix(r, doubleArrayOf(1.0, 0.0, -1.0, -2.0, -3.0))
        val es = EigenDecomposition(m).getSolver()
        assertFalse(es.isNonSingular())
        try {
            es.getInverse()
            fail("an exception should have been thrown")
        } catch (ime: Throwable) {
            // expected SingularMatrixException
        }
    }

    /** test invertible matrix  */
    @Test
    fun testInvertible() {
        val r = Random(9994100315208L)
        val m: RealMatrix =
            EigenDecompositionTest.createTestMatrix(r, doubleArrayOf(1.0, 0.5, -1.0, -2.0, -3.0))
        val es = EigenDecomposition(m).getSolver()
        assertTrue(es.isNonSingular())
        val inverse: RealMatrix = es.getInverse()
        val error: RealMatrix = mk.linalg.dot(m, inverse) - mk.identity(m.getRowDimension())
        assertEquals(0.0, error.getNorm(), 4.0e-15)
    }

    /**
     * Verifies operation on very small values.
     * Matrix with eigenvalues {8e-100, -1e-100, -1e-100}
     */
    @Test
    fun testInvertibleTinyValues() {
        val tiny = 1e-100
        var m: RealMatrix = arrayOf(
            doubleArrayOf(3.0, 2.0, 4.0),
            doubleArrayOf(2.0, 0.0, 2.0),
            doubleArrayOf(4.0, 2.0, 3.0)
        ).toNDArray()
        m *= tiny
        val ed = EigenDecomposition(m)
        val inv: RealMatrix = ed.getSolver().getInverse()
        val id: RealMatrix = mk.linalg.dot(m, inv)
        for (i in 0 until m.getRowDimension()) {
            for (j in 0 until m.getColumnDimension()) {
                if (i == j) {
                    assertTrue(Precision.equals(1.0, id[i][j], 1e-15))
                } else {
                    assertTrue(Precision.equals(0.0, id[i][j], 1e-15))
                }
            }
        }
    }

    @Test
    fun testNonInvertibleMath1045() {
        val eigen = EigenDecomposition(bigSingular.toNDArray())
        try {
            eigen.getSolver().getInverse()
            fail("expected SingularMatrixException")
        } catch (t: Throwable) {
            // expected SingularMatrixException
        }

    }

    @Test
    fun testZeroMatrix() {
        val eigen = EigenDecomposition(arrayOf(doubleArrayOf(0.0)).toNDArray())
        try {
            eigen.getSolver().getInverse()
            fail("expected SingularMatrixException")
        } catch (t: Throwable) {
            // expected SingularMatrixException
        }

    }

    @Test
    fun testIsNonSingularTinyOutOfOrderEigenvalue() {
        val eigen = EigenDecomposition(
            arrayOf(
                doubleArrayOf(1e-13, 0.0),
                doubleArrayOf(1.0, 1.0)
            ).toNDArray()
        )
        assertFalse(eigen.getSolver().isNonSingular(), "Singular matrix not detected")
    }

    /** test solve dimension errors  */
    @Test
    fun testSolveDimensionErrors() {
        val refValues = doubleArrayOf(2.003, 2.002, 2.001, 1.001, 1.000, 0.001)
        val matrix: RealMatrix =
            EigenDecompositionTest.createTestMatrix(Random(35992629946426L), refValues)
        val es = EigenDecomposition(matrix).getSolver()
        val b: RealMatrix = Array(2) { DoubleArray(2) }.toNDArray()
        try {
            es.solve(b)
            fail("an exception should have been thrown")
        } catch (iae: Throwable) {
            // expected MathIllegalArgumentException
        }
        try {
            es.solve(b.getColumnVector(0))
            fail("an exception should have been thrown")
        } catch (iae: Throwable) {
            // expected MathIllegalArgumentException
        }
    }

    /** test solve  */
    @Test
    fun testSolve() {
        val m: RealMatrix = arrayOf(
            doubleArrayOf(91.0, 5.0, 29.0, 32.0, 40.0, 14.0),
            doubleArrayOf(5.0, 34.0, -1.0, 0.0, 2.0, -1.0),
            doubleArrayOf(29.0, -1.0, 12.0, 9.0, 21.0, 8.0),
            doubleArrayOf(32.0, 0.0, 9.0, 14.0, 9.0, 0.0),
            doubleArrayOf(40.0, 2.0, 21.0, 9.0, 51.0, 19.0),
            doubleArrayOf(14.0, -1.0, 8.0, 0.0, 19.0, 14.0)
        ).toNDArray()
        val es = EigenDecomposition(m).getSolver()
        val b: RealMatrix = arrayOf(
            doubleArrayOf(1561.0, 269.0, 188.0),
            doubleArrayOf(69.0, -21.0, 70.0),
            doubleArrayOf(739.0, 108.0, 63.0),
            doubleArrayOf(324.0, 86.0, 59.0),
            doubleArrayOf(1624.0, 194.0, 107.0),
            doubleArrayOf(796.0, 69.0, 36.0)
        ).toNDArray()
        val xRef: RealMatrix = arrayOf(
            doubleArrayOf(1.0, 2.0, 1.0),
            doubleArrayOf(2.0, -1.0, 2.0),
            doubleArrayOf(4.0, 2.0, 3.0),
            doubleArrayOf(8.0, -1.0, 0.0),
            doubleArrayOf(16.0, 2.0, 0.0),
            doubleArrayOf(32.0, -1.0, 0.0)
        ).toNDArray()

        // using RealMatrix
        val solution: RealMatrix = es.solve(b)
        assertEquals(0.0, (solution - xRef).getNorm(), 2.5e-12)

        // using RealVector
        for (i in 0 until b.getColumnDimension()) {
            assertEquals(
                0.0,
                (es.solve(b.getColumnVector(i)) - (xRef.getColumnVector(i))).getNorm(),
                2.0e-11
            )
        }
    }

}