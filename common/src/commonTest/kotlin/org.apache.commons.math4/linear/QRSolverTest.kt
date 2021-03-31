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
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlin.test.fail

class QRSolverTest {
    var testData3x3NonSingular: Array<DoubleArray> = arrayOf(
        doubleArrayOf(12.0, -51.0, 4.0),
        doubleArrayOf(6.0, 167.0, -68.0),
        doubleArrayOf(-4.0, 24.0, -41.0)
    )
    var testData3x3Singular: Array<DoubleArray> = arrayOf(
        doubleArrayOf(1.0, 2.0, 2.0),
        doubleArrayOf(2.0, 4.0, 6.0),
        doubleArrayOf(4.0, 8.0, 12.0)
    )
    var testData3x4: Array<DoubleArray> = arrayOf(
        doubleArrayOf(12.0, -51.0, 4.0, 1.0),
        doubleArrayOf(6.0, 167.0, -68.0, 2.0),
        doubleArrayOf(-4.0, 24.0, -41.0, 3.0)
    )
    var testData4x3: Array<DoubleArray> = arrayOf(
        doubleArrayOf(12.0, -51.0, 4.0),
        doubleArrayOf(6.0, 167.0, -68.0),
        doubleArrayOf(-4.0, 24.0, -41.0),
        doubleArrayOf(-5.0, 34.0, 7.0)
    )

    /** test rank  */
    @Test
    fun testRank() {
        var solver = QRDecomposition(testData3x3NonSingular.toNDArray()).getSolver()
        assertTrue(solver.isNonSingular())
        solver = QRDecomposition(testData3x3Singular.toNDArray()).getSolver()
        assertFalse(solver.isNonSingular())
        solver = QRDecomposition(testData3x4.toNDArray()).getSolver()
        assertTrue(solver.isNonSingular())
        solver = QRDecomposition(testData4x3.toNDArray()).getSolver()
        assertTrue(solver.isNonSingular())
    }

    /** test solve dimension errors  */
    @Test
    fun testSolveDimensionErrors() {
        val solver = QRDecomposition(testData3x3NonSingular.toNDArray()).getSolver()
        val b: RealMatrix = Array(2) { DoubleArray(2) }.toNDArray()
        try {
            solver.solve(b)
            fail("an exception should have been thrown")
        } catch (iae: Throwable) {
            // expected behavior
        }
        try {
            solver.solve(b.getColumnVector(0))
            fail("an exception should have been thrown")
        } catch (iae: Throwable) {
            // expected behavior
        }
    }

    /** test solve rank errors  */
    @Test
    fun testSolveRankErrors() {
        val solver = QRDecomposition(testData3x3Singular.toNDArray()).getSolver()
        val b: RealMatrix = Array(3) { DoubleArray(2) }.toNDArray()
        try {
            solver.solve(b)
            fail("an exception should have been thrown")
        } catch (iae: Throwable) {
            // expected behavior
        }
        try {
            solver.solve(b.getColumnVector(0))
            fail("an exception should have been thrown")
        } catch (iae: Throwable) {
            // expected behavior
        }
    }

    /** test solve  */
    @Test
    fun testSolve() {
        val decomposition = QRDecomposition(testData3x3NonSingular.toNDArray())
        val solver = decomposition.getSolver()
        val b: RealMatrix = arrayOf(
            doubleArrayOf(-102.0, 12250.0),
            doubleArrayOf(544.0, 24500.0),
            doubleArrayOf(167.0, -36750.0)
        ).toNDArray()
        val xRef: RealMatrix = arrayOf(
            doubleArrayOf(1.0, 2515.0),
            doubleArrayOf(2.0, 422.0),
            doubleArrayOf(-3.0, 898.0)
        ).toNDArray()

        // using RealMatrix
        assertEquals(0.0, (solver.solve(b) - xRef).getNorm(), 2.0e-16 * xRef.getNorm())

        // using ArrayRealVector
        for (i in 0 until b.getColumnDimension()) {
            val x: RealVector = solver.solve(b.getColumnVector(i))
            val error: Double = (x - (xRef.getColumnVector(i))).getNorm()
            assertEquals(0.0, error, 3.0e-16 * xRef.getColumnVector(i).getNorm())
        }
    }

    @Test
    fun testOverdetermined() {
        val r = Random(5559252868205245L)
        val p: Int = 7 * BLOCK_SIZE / 4
        val q: Int = 5 * BLOCK_SIZE / 4
        val a: RealMatrix = createTestMatrix(r, p, q)
        val xRef: RealMatrix = createTestMatrix(r, q, BLOCK_SIZE + 3)

        // build a perturbed system: A.X + noise = B
        var b: RealMatrix = mk.linalg.dot(a, xRef)
        val noise = 0.001
        b = b.mapInOrder { _: Int, _: Int, value: Double ->
            value * (1.0 + noise * (2 * r.nextDouble() - 1))
        }

        // despite perturbation, the least square solution should be pretty good
        val x: RealMatrix = QRDecomposition(a).getSolver().solve(b)
        assertEquals(0.0, (x - xRef).getNorm(), 0.01 * noise * p * q)
    }

    @Test
    fun testUnderdetermined() {
        val r = Random(42185006424567123L)
        val p: Int = 5 * BLOCK_SIZE / 4
        val q: Int = 7 * BLOCK_SIZE / 4
        val a: RealMatrix = createTestMatrix(r, p, q)
        val xRef: RealMatrix = createTestMatrix(r, q, BLOCK_SIZE + 3)
        val b: RealMatrix = mk.linalg.dot(a, xRef)
        val x: RealMatrix = QRDecomposition(a).getSolver().solve(b)

        // too many equations, the system cannot be solved at all
        assertTrue((x - xRef).getNorm() / (p * q) > 0.01)

        // the last unknown should have been set to 0
        assertEquals(
            0.0,
            x.getSubMatrix(p, q - 1, 0, x.getColumnDimension() - 1).getNorm(),
            0.0
        )
    }

    private fun createTestMatrix(r: Random, rows: Int, columns: Int): RealMatrix {
        return Array(rows) { DoubleArray(columns) { 2.0 * r.nextDouble() - 1.0 } }.toNDArray()
    }

    companion object {
        fun RealMatrix.mapInOrder(visit: (row: Int, column: Int, value: Double) -> Double): RealMatrix {
            return getData().let {
                val rows = getRowDimension()
                val columns = getColumnDimension()
                for (row in 0 until rows) {
                    for (column in 0 until columns) {
                        val oldValue = this[row][column]
                        val newValue = visit(row, column, oldValue)
                        it[row][column] = newValue
                    }
                }
                it.toNDArray()
            }
        }

        const val BLOCK_SIZE = 52
    }

}