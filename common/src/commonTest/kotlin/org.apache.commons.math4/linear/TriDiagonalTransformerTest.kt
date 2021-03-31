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
import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertSame
import kotlin.test.fail

class TriDiagonalTransformerTest {
    private val testSquare5: Array<DoubleArray> = arrayOf(
        doubleArrayOf(1.0, 2.0, 3.0, 1.0, 1.0),
        doubleArrayOf(2.0, 1.0, 1.0, 3.0, 1.0),
        doubleArrayOf(3.0, 1.0, 1.0, 1.0, 2.0),
        doubleArrayOf(1.0, 3.0, 1.0, 2.0, 1.0),
        doubleArrayOf(1.0, 1.0, 2.0, 1.0, 3.0)
    )
    private val testSquare3: Array<DoubleArray> = arrayOf(
        doubleArrayOf(1.0, 3.0, 4.0),
        doubleArrayOf(3.0, 2.0, 2.0),
        doubleArrayOf(4.0, 2.0, 0.0)
    )

    @Test
    fun testNonSquare() {
        try {
            TriDiagonalTransformer(Array(3) { DoubleArray(2) }.toNDArray())
            fail("an exception should have been thrown")
        } catch (ime: Throwable) {
            // expected behavior
        }
    }

    @Test
    fun testAEqualQTQt() {
        checkAEqualQTQt(testSquare5.toNDArray())
        checkAEqualQTQt(testSquare3.toNDArray())
    }

    private fun checkAEqualQTQt(matrix: RealMatrix) {
        val transformer = TriDiagonalTransformer(matrix)
        val q: RealMatrix = transformer.getQ()
        val qT: RealMatrix = transformer.getQT()
        val t: RealMatrix = transformer.getT()
        val norm: Double = (mk.linalg.dot(mk.linalg.dot(q, t), qT) - matrix).getNorm()
        assertEquals(0.0, norm, 4.0e-15)
    }

    @Test
    fun testNoAccessBelowDiagonal() {
        checkNoAccessBelowDiagonal(testSquare5)
        checkNoAccessBelowDiagonal(testSquare3)
    }

    private fun checkNoAccessBelowDiagonal(data: Array<DoubleArray>) {
        val modifiedData = Array(data.size) {
            data[it].copyOf().apply { fill(Double.NaN, 0, it) }
        }
        val matrix: RealMatrix = modifiedData.toNDArray()
        val transformer = TriDiagonalTransformer(matrix)
        val q: RealMatrix = transformer.getQ()
        val qT: RealMatrix = transformer.getQT()
        val t: RealMatrix = transformer.getT()
        val norm: Double = (mk.linalg.dot(mk.linalg.dot(q, t), qT) - data.toNDArray()).getNorm()
        assertEquals(0.0, norm, 4.0e-15)
    }

    @Test
    fun testQOrthogonal() {
        checkOrthogonal(TriDiagonalTransformer(testSquare5.toNDArray()).getQ())
        checkOrthogonal(TriDiagonalTransformer(testSquare3.toNDArray()).getQ())
    }

    @Test
    fun testQTOrthogonal() {
        checkOrthogonal(TriDiagonalTransformer(testSquare5.toNDArray()).getQT())
        checkOrthogonal(TriDiagonalTransformer(testSquare3.toNDArray()).getQT())
    }

    private fun checkOrthogonal(m: RealMatrix) {
        val mTm: RealMatrix = mk.linalg.dot(m.transpose(), m)
        val id: RealMatrix = mk.identity(mTm.getRowDimension())
        assertEquals(0.0, (mTm - id).getNorm(), 1.0e-15)
    }

    @Test
    fun testTTriDiagonal() {
        checkTriDiagonal(TriDiagonalTransformer(testSquare5.toNDArray()).getT())
        checkTriDiagonal(TriDiagonalTransformer(testSquare3.toNDArray()).getT())
    }

    private fun checkTriDiagonal(m: RealMatrix) {
        val rows: Int = m.getRowDimension()
        val cols: Int = m.getColumnDimension()
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                if (i < j - 1 || i > j + 1) {
                    assertEquals(0.0, m[i][j], 1.0e-16)
                }
            }
        }
    }

    @Test
    fun testMatricesValues5() {
        checkMatricesValues(
            testSquare5,
            arrayOf(
                doubleArrayOf(1.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(
                    0.0,
                    -0.5163977794943222,
                    0.016748280772542083,
                    0.839800693771262,
                    0.16669620021405473
                ),
                doubleArrayOf(
                    0.0,
                    -0.7745966692414833,
                    -0.4354553000860955,
                    -0.44989322880603355,
                    -0.08930153582895772
                ),
                doubleArrayOf(
                    0.0,
                    -0.2581988897471611,
                    0.6364346693566014,
                    -0.30263204032131164,
                    0.6608313651342882
                ),
                doubleArrayOf(
                    0.0,
                    -0.2581988897471611,
                    0.6364346693566009,
                    -0.027289660803112598,
                    -0.7263191580755246
                )
            ),
            doubleArrayOf(1.0, 4.4, 1.433099579242636, -0.89537362758743, 2.062274048344794),
            doubleArrayOf(-sqrt(15.0), -3.0832882879592476, 0.6082710842351517, 1.1786086405912128)
        )
    }

    @Test
    fun testMatricesValues3() {
        checkMatricesValues(
            testSquare3,
            arrayOf(
                doubleArrayOf(1.0, 0.0, 0.0),
                doubleArrayOf(0.0, -0.6, 0.8),
                doubleArrayOf(0.0, -0.8, -0.6)
            ),
            doubleArrayOf(1.0, 2.64, -0.64),
            doubleArrayOf(-5.0, -1.52)
        )
    }

    private fun checkMatricesValues(
        matrix: Array<DoubleArray>,
        qRef: Array<DoubleArray>,
        mainDiagnonal: DoubleArray,
        secondaryDiagonal: DoubleArray
    ) {
        val transformer = TriDiagonalTransformer(matrix.toNDArray())

        // check values against known references
        val q: RealMatrix = transformer.getQ()
        assertEquals(0.0, (q - qRef.toNDArray()).getNorm(), 1.0e-14)
        val t: RealMatrix = transformer.getT()
        val tData = Array(mainDiagnonal.size) { DoubleArray(mainDiagnonal.size) }
        for (i in mainDiagnonal.indices) {
            tData[i][i] = mainDiagnonal[i]
            if (i > 0) {
                tData[i][i - 1] = secondaryDiagonal[i - 1]
            }
            if (i < secondaryDiagonal.size) {
                tData[i][i + 1] = secondaryDiagonal[i]
            }
        }
        assertEquals(0.0, (t - tData.toNDArray()).getNorm(), 1.0e-14)

        // check the same cached instance is returned the second time
        assertSame(q, transformer.getQ())
        assertSame(t, transformer.getT())
    }

}