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
import org.apache.commons.math4.linear.QRSolverTest.Companion.BLOCK_SIZE
import org.apache.commons.math4.linear.QRSolverTest.Companion.mapInOrder
import org.jetbrains.kotlinx.multik.api.empty
import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertSame
import kotlin.test.fail

class QRDecompositionTest {
    private val testData3x3NonSingular: Array<DoubleArray> = arrayOf(
        doubleArrayOf(12.0, -51.0, 4.0),
        doubleArrayOf(6.0, 167.0, -68.0),
        doubleArrayOf(-4.0, 24.0, -41.0)
    )
    private val testData3x3Singular: Array<DoubleArray> = arrayOf(
        doubleArrayOf(1.0, 4.0, 7.0),
        doubleArrayOf(2.0, 5.0, 8.0),
        doubleArrayOf(3.0, 6.0, 9.0)
    )
    private val testData3x4: Array<DoubleArray> = arrayOf(
        doubleArrayOf(12.0, -51.0, 4.0, 1.0),
        doubleArrayOf(6.0, 167.0, -68.0, 2.0),
        doubleArrayOf(-4.0, 24.0, -41.0, 3.0)
    )
    private val testData4x3: Array<DoubleArray> = arrayOf(
        doubleArrayOf(12.0, -51.0, 4.0),
        doubleArrayOf(6.0, 167.0, -68.0),
        doubleArrayOf(-4.0, 24.0, -41.0),
        doubleArrayOf(-5.0, 34.0, 7.0)
    )

    /** test dimensions  */
    @Test
    fun testDimensions() {
        checkDimension(testData3x3NonSingular.toNDArray())
        checkDimension(testData4x3.toNDArray())
        checkDimension(testData3x4.toNDArray())
        val r = Random(643895747384642L)
        val p: Int = 5 * BLOCK_SIZE / 4
        val q: Int = 7 * BLOCK_SIZE / 4
        checkDimension(createTestMatrix(r, p, q))
        checkDimension(createTestMatrix(r, q, p))
    }

    private fun checkDimension(m: RealMatrix) {
        val rows: Int = m.getRowDimension()
        val columns: Int = m.getColumnDimension()
        val qr = QRDecomposition(m)
        assertEquals(rows, qr.getQ().getRowDimension())
        assertEquals(rows, qr.getQ().getColumnDimension())
        assertEquals(rows, qr.getR().getRowDimension())
        assertEquals(columns, qr.getR().getColumnDimension())
    }

    /** test A = QR  */
    @Test
    fun testAEqualQR() {
        checkAEqualQR(testData3x3NonSingular.toNDArray())
        checkAEqualQR(testData3x3Singular.toNDArray())
        checkAEqualQR(testData3x4.toNDArray())
        checkAEqualQR(testData4x3.toNDArray())
        val r = Random(643895747384642L)
        val p: Int = 5 * BLOCK_SIZE / 4
        val q: Int = 7 * BLOCK_SIZE / 4
        checkAEqualQR(createTestMatrix(r, p, q))
        checkAEqualQR(createTestMatrix(r, q, p))
    }

    private fun checkAEqualQR(m: RealMatrix) {
        val qr = QRDecomposition(m)
        val norm: Double = (mk.linalg.dot(qr.getQ(), qr.getR()) - (m)).getNorm()
        assertEquals(0.0, norm, normTolerance)
    }

    /** test the orthogonality of Q  */
    @Test
    fun testQOrthogonal() {
        checkQOrthogonal(testData3x3NonSingular.toNDArray())
        checkQOrthogonal(testData3x3Singular.toNDArray())
        checkQOrthogonal(testData3x4.toNDArray())
        checkQOrthogonal(testData4x3.toNDArray())
        val r = Random(643895747384642L)
        val p: Int = 5 * BLOCK_SIZE / 4
        val q: Int = 7 * BLOCK_SIZE / 4
        checkQOrthogonal(createTestMatrix(r, p, q))
        checkQOrthogonal(createTestMatrix(r, q, p))
    }

    private fun checkQOrthogonal(m: RealMatrix) {
        val qr = QRDecomposition(m)
        val eye: RealMatrix = mk.identity(m.getRowDimension())
        val norm: Double = (mk.linalg.dot(qr.getQT(), qr.getQ()) - eye).getNorm()
        assertEquals(0.0, norm, normTolerance)
    }

    /** test that R is upper triangular  */
    @Test
    fun testRUpperTriangular() {
        var matrix: RealMatrix = testData3x3NonSingular.toNDArray()
        checkUpperTriangular(QRDecomposition(matrix).getR())
        matrix = testData3x3Singular.toNDArray()
        checkUpperTriangular(QRDecomposition(matrix).getR())
        matrix = testData3x4.toNDArray()
        checkUpperTriangular(QRDecomposition(matrix).getR())
        matrix = testData4x3.toNDArray()
        checkUpperTriangular(QRDecomposition(matrix).getR())
        val r = Random(643895747384642L)
        val p: Int = 5 * BLOCK_SIZE / 4
        val q: Int = 7 * BLOCK_SIZE / 4
        matrix = createTestMatrix(r, p, q)
        checkUpperTriangular(QRDecomposition(matrix).getR())
        matrix = createTestMatrix(r, p, q)
        checkUpperTriangular(QRDecomposition(matrix).getR())
    }

    private fun checkUpperTriangular(m: RealMatrix) {
        m.mapInOrder { row: Int, column: Int, value: Double ->
            if (column < row) {
                assertEquals(0.0, value, entryTolerance)
            }
            value
        }
    }

    /** test that H is trapezoidal  */
    @Test
    fun testHTrapezoidal() {
        var matrix: RealMatrix = testData3x3NonSingular.toNDArray()
        checkTrapezoidal(QRDecomposition(matrix).getH())
        matrix = testData3x3Singular.toNDArray()
        checkTrapezoidal(QRDecomposition(matrix).getH())
        matrix = testData3x4.toNDArray()
        checkTrapezoidal(QRDecomposition(matrix).getH())
        matrix = testData4x3.toNDArray()
        checkTrapezoidal(QRDecomposition(matrix).getH())
        val r = Random(643895747384642L)
        val p: Int = 5 * BLOCK_SIZE / 4
        val q: Int = 7 * BLOCK_SIZE / 4
        matrix = createTestMatrix(r, p, q)
        checkTrapezoidal(QRDecomposition(matrix).getH())
        matrix = createTestMatrix(r, p, q)
        checkTrapezoidal(QRDecomposition(matrix).getH())
    }

    private fun checkTrapezoidal(m: RealMatrix) {
        m.mapInOrder { row: Int, column: Int, value: Double ->
            if (column > row) {
                assertEquals(0.0, value, entryTolerance)
            }
            value
        }
    }

    /** test matrices values  */
    @Test
    fun testMatricesValues() {
        val qr = QRDecomposition(testData3x3NonSingular.toNDArray())
        val qRef: RealMatrix = arrayOf(
            doubleArrayOf(-12.0 / 14.0, 69.0 / 175.0, -58.0 / 175.0),
            doubleArrayOf(-6.0 / 14.0, -158.0 / 175.0, 6.0 / 175.0),
            doubleArrayOf(4.0 / 14.0, -30.0 / 175.0, -165.0 / 175.0)
        ).toNDArray()
        val rRef: RealMatrix = arrayOf(
            doubleArrayOf(-14.0, -21.0, 14.0),
            doubleArrayOf(0.0, -175.0, 70.0),
            doubleArrayOf(0.0, 0.0, 35.0)
        ).toNDArray()
        val hRef: RealMatrix = arrayOf(
            doubleArrayOf(26.0 / 14.0, 0.0, 0.0),
            doubleArrayOf(6.0 / 14.0, 648.0 / 325.0, 0.0),
            doubleArrayOf(-4.0 / 14.0, 36.0 / 325.0, 2.0)
        ).toNDArray()

        // check values against known references
        val q: RealMatrix = qr.getQ()
        assertEquals(0.0, (q - qRef).getNorm(), 1.0e-13)
        val qT: RealMatrix = qr.getQT()
        assertEquals(0.0, (qT - qRef.transpose()).getNorm(), 1.0e-13)
        val r: RealMatrix = qr.getR()
        assertEquals(0.0, (r - rRef).getNorm(), 1.0e-13)
        val h: RealMatrix = qr.getH()
        assertEquals(0.0, (h - hRef).getNorm(), 1.0e-13)

        // check the same cached instance is returned the second time
        assertSame(q, qr.getQ())
        assertSame(r, qr.getR())
        assertSame(h, qr.getH())
    }

    @Test
    fun testNonInvertible() {
        val qr = QRDecomposition(testData3x3Singular.toNDArray())
        try {
            qr.getSolver().getInverse()
            fail("Expected SingularMatrixException")
        } catch (t: Throwable) {
            // Expected SingularMatrixException
        }

    }

    @Test
    fun testInvertTallSkinny() {
        val a: RealMatrix = testData4x3.toNDArray()
        val pinv: RealMatrix = QRDecomposition(a).getSolver().getInverse()
        assertEquals(0.0, (mk.linalg.dot(pinv, a) - mk.identity(3)).getNorm(), 1.0e-6)
    }

    @Test
    fun testInvertShortWide() {
        val a: RealMatrix = testData3x4.toNDArray()
        val pinv: RealMatrix = QRDecomposition(a).getSolver().getInverse()
        assertEquals(0.0, (mk.linalg.dot(a, pinv) - mk.identity(3)).getNorm(), 1.0e-6)
        assertEquals(
            0.0,
            (mk.linalg.dot(pinv, a).getSubMatrix(0, 2, 0, 2) - mk.identity(3)).getNorm(),
            1.0e-6
        )
    }

    private fun createTestMatrix(r: Random, rows: Int, columns: Int): RealMatrix {
        val m: RealMatrix = mk.empty(rows, columns)
        m.mapInOrder { _: Int, _: Int, _: Double ->
            2.0 * r.nextDouble() - 1.0
        }
        return m
    }

    @Test
    fun testQRSingular() {
        val a: RealMatrix = arrayOf(
            doubleArrayOf(1.0, 6.0, 4.0),
            doubleArrayOf(2.0, 4.0, -1.0),
            doubleArrayOf(-1.0, 2.0, 5.0)
        ).toNDArray()
        val b: RealVector = doubleArrayOf(5.0, 6.0, 1.0).toNDArray()
        try {
            QRDecomposition(a, 1.0e-15).getSolver().solve(b)
            fail("Expected SingularMatrixException")
        } catch (t: Throwable) {
            // Expected SingularMatrixException
        }
    }

    companion object {
        private const val entryTolerance = 10e-16
        private const val normTolerance = 10e-14
    }

}