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
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertSame
import kotlin.test.fail

class HessenbergTransformerTest {
    private val testSquare5: Array<DoubleArray> = arrayOf(
        doubleArrayOf(5.0, 4.0, 3.0, 2.0, 1.0),
        doubleArrayOf(1.0, 4.0, 0.0, 3.0, 3.0),
        doubleArrayOf(2.0, 0.0, 3.0, 0.0, 0.0),
        doubleArrayOf(3.0, 2.0, 1.0, 2.0, 5.0),
        doubleArrayOf(4.0, 2.0, 1.0, 4.0, 1.0)
    )
    private val testSquare3: Array<DoubleArray> = arrayOf(
        doubleArrayOf(2.0, -1.0, 1.0),
        doubleArrayOf(-1.0, 2.0, 1.0),
        doubleArrayOf(1.0, -1.0, 2.0)
    )

    // from http://eigen.tuxfamily.org/dox/classEigen_1_1HessenbergDecomposition.html
    private val testRandom: Array<DoubleArray> = arrayOf(
        doubleArrayOf(0.680, 0.823, -0.4440, -0.2700),
        doubleArrayOf(-0.211, -0.605, 0.1080, 0.0268),
        doubleArrayOf(0.566, -0.330, -0.0452, 0.9040),
        doubleArrayOf(0.597, 0.536, 0.2580, 0.8320)
    )

    @Test
    fun testNonSquare() {
        try {
            HessenbergTransformer(Array(3) { DoubleArray(2) }.toNDArray())
            fail("an exception should have been thrown")
        } catch (ime: Throwable) {
            // expected NonSquareMatrixException
        }
    }

    @Test
    fun testAEqualPHPt() {
        checkAEqualPHPt(testSquare5.toNDArray())
        checkAEqualPHPt(testSquare3.toNDArray())
        checkAEqualPHPt(testRandom.toNDArray())
    }

    @Test
    fun testPOrthogonal() {
        checkOrthogonal(HessenbergTransformer(testSquare5.toNDArray()).getP())
        checkOrthogonal(HessenbergTransformer(testSquare3.toNDArray()).getP())
    }

    @Test
    fun testPTOrthogonal() {
        checkOrthogonal(HessenbergTransformer(testSquare5.toNDArray()).getPT())
        checkOrthogonal(HessenbergTransformer(testSquare3.toNDArray()).getPT())
    }

    @Test
    fun testHessenbergForm() {
        checkHessenbergForm(HessenbergTransformer(testSquare5.toNDArray()).getH())
        checkHessenbergForm(HessenbergTransformer(testSquare3.toNDArray()).getH())
    }

    @Test
    fun testRandomData() {
        for (run in 0..99) {
            val r = Random

            // matrix size
            val size = r.nextInt(20) + 4
            val data = Array(size) { DoubleArray(size) }
            for (i in 0 until size) {
                for (j in 0 until size) {
                    data[i][j] = r.nextInt(100).toDouble()
                }
            }
            val m: RealMatrix = data.toNDArray()
            val h: RealMatrix = checkAEqualPHPt(m)
            checkHessenbergForm(h)
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
                    -0.182574185835055,
                    0.784218758628863,
                    0.395029040913988,
                    -0.442289115981669
                ),
                doubleArrayOf(
                    0.0,
                    -0.365148371670111,
                    -0.337950625265477,
                    -0.374110794088820,
                    -0.782621974707823
                ),
                doubleArrayOf(
                    0.0,
                    -0.547722557505166,
                    0.402941130124223,
                    -0.626468266309003,
                    0.381019628053472
                ),
                doubleArrayOf(
                    0.0,
                    -0.730296743340221,
                    -0.329285224617644,
                    0.558149336547665,
                    0.216118545309225
                )
            ),
            arrayOf(
                doubleArrayOf(
                    5.0,
                    -3.65148371670111,
                    2.59962019434982,
                    -0.237003414680848,
                    -3.13886458663398
                ),
                doubleArrayOf(
                    -5.47722557505166,
                    6.9,
                    -2.29164066120599,
                    0.207283564429169,
                    0.703858369151728
                ),
                doubleArrayOf(
                    0.0,
                    -4.21386600008432,
                    2.30555659846067,
                    2.74935928725112,
                    0.857569835914113
                ),
                doubleArrayOf(0.0, 0.0, 2.86406180891882, -1.11582249161595, 0.817995267184158),
                doubleArrayOf(0.0, 0.0, 0.0, 0.683518597386085, 1.91026589315528)
            )
        )
    }

    @Test
    fun testMatricesValues3() {
        checkMatricesValues(
            testSquare3,
            arrayOf(
                doubleArrayOf(1.0, 0.0, 0.0),
                doubleArrayOf(0.0, -0.707106781186547, 0.707106781186547),
                doubleArrayOf(0.0, 0.707106781186547, 0.707106781186548)
            ),
            arrayOf(
                doubleArrayOf(2.0, 1.41421356237309, 0.0),
                doubleArrayOf(1.41421356237310, 2.0, -1.0),
                doubleArrayOf(0.0, 1.0, 2.0)
            )
        )
    }

    ///////////////////////////////////////////////////////////////////////////
    // Test helpers
    ///////////////////////////////////////////////////////////////////////////
    private fun checkAEqualPHPt(matrix: RealMatrix): RealMatrix {
        val transformer = HessenbergTransformer(matrix)
        val p: RealMatrix = transformer.getP()
        val pT: RealMatrix = transformer.getPT()
        val h: RealMatrix = transformer.getH()
        val result: RealMatrix = mk.linalg.dot(mk.linalg.dot(p, h), pT)
        val norm: Double = (result - matrix).getNorm()
        assertEquals(0.0, norm, 1.0e-10)
        for (i in 0 until matrix.getRowDimension()) {
            for (j in 0 until matrix.getColumnDimension()) {
                if (i > j + 1) {
                    assertEquals(matrix[i][j], result[i][j], 1.0e-12)
                }
            }
        }
        return transformer.getH()
    }

    private fun checkOrthogonal(m: RealMatrix) {
        val mTm: RealMatrix = mk.linalg.dot(m.transpose(), m)
        val id: RealMatrix = mk.identity(mTm.getRowDimension())
        assertEquals(0.0, (mTm - id).getNorm(), 1.0e-14)
    }

    private fun checkHessenbergForm(m: RealMatrix) {
        val rows: Int = m.getRowDimension()
        val cols: Int = m.getColumnDimension()
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                if (i > j + 1) {
                    assertEquals(0.0, m[i][j], 1.0e-16)
                }
            }
        }
    }

    private fun checkMatricesValues(
        matrix: Array<DoubleArray>,
        pRef: Array<DoubleArray>,
        hRef: Array<DoubleArray>
    ) {
        val transformer = HessenbergTransformer(matrix.toNDArray())

        // check values against known references
        val p: RealMatrix = transformer.getP()
        assertEquals(0.0, (p - pRef.toNDArray()).getNorm(), 1.0e-14)
        val h: RealMatrix = transformer.getH()
        assertEquals(0.0, (h - hRef.toNDArray()).getNorm(), 1.0e-14)

        // check the same cached instance is returned the second time
        assertSame(p, transformer.getP())
        assertSame(h, transformer.getH())
    }
}