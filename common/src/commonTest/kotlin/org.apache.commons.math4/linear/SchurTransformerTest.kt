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
import kotlin.test.fail

class SchurTransformerTest {
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

    // from http://eigen.tuxfamily.org/dox/classEigen_1_1RealSchur.html
    private val testRandom: Array<DoubleArray> = arrayOf(
        doubleArrayOf(0.680, -0.3300, -0.2700, -0.717, -0.687, 0.0259),
        doubleArrayOf(-0.211, 0.5360, 0.0268, 0.214, -0.198, 0.6780),
        doubleArrayOf(0.566, -0.4440, 0.9040, -0.967, -0.740, 0.2250),
        doubleArrayOf(0.597, 0.1080, 0.8320, -0.514, -0.782, -0.4080),
        doubleArrayOf(0.823, -0.0452, 0.2710, -0.726, 0.998, 0.2750),
        doubleArrayOf(-0.605, 0.2580, 0.4350, 0.608, -0.563, 0.0486)
    )

    @Test
    fun testNonSquare() {
        try {
            SchurTransformer(Array(3) { DoubleArray(2) }.toNDArray())
            fail("an exception should have been thrown")
        } catch (ime: Throwable) {
            // expected behavior
        }
    }

    @Test
    fun testAEqualPTPt() {
        checkAEqualPTPt(testSquare5.toNDArray())
        checkAEqualPTPt(testSquare3.toNDArray())
        checkAEqualPTPt(testRandom.toNDArray())
    }

    @Test
    fun testPOrthogonal() {
        checkOrthogonal(SchurTransformer(testSquare5.toNDArray()).getP())
        checkOrthogonal(SchurTransformer(testSquare3.toNDArray()).getP())
        checkOrthogonal(SchurTransformer(testRandom.toNDArray()).getP())
    }

    @Test
    fun testPTOrthogonal() {
        checkOrthogonal(SchurTransformer(testSquare5.toNDArray()).getPT())
        checkOrthogonal(SchurTransformer(testSquare3.toNDArray()).getPT())
        checkOrthogonal(SchurTransformer(testRandom.toNDArray()).getPT())
    }

    @Test
    fun testSchurForm() {
        checkSchurForm(SchurTransformer(testSquare5.toNDArray()).getT())
        checkSchurForm(SchurTransformer(testSquare3.toNDArray()).getT())
        checkSchurForm(SchurTransformer(testRandom.toNDArray()).getT())
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
            val s: RealMatrix = checkAEqualPTPt(m)
            checkSchurForm(s)
        }
    }

    @Test
    fun testMath848() {
        val data = arrayOf(
            doubleArrayOf(
                0.1849449280,
                -0.0646971046,
                0.0774755812,
                -0.0969651755,
                -0.0692648806,
                0.3282344352,
                -0.0177423074,
                0.2063136340
            ),
            doubleArrayOf(
                -0.0742700134,
                -0.0289063030,
                -0.0017269460,
                -0.0375550146,
                -0.0487737922,
                -0.2616837868,
                -0.0821201295,
                -0.2530000167
            ),
            doubleArrayOf(
                0.2549910127,
                0.0995733692,
                -0.0009718388,
                0.0149282808,
                0.1791878897,
                -0.0823182816,
                0.0582629256,
                0.3219545182
            ),
            doubleArrayOf(
                -0.0694747557,
                -0.1880649148,
                -0.2740630911,
                0.0720096468,
                -0.1800836914,
                -0.3518996425,
                0.2486747833,
                0.6257938167
            ),
            doubleArrayOf(
                0.0536360918,
                -0.1339297778,
                0.2241579764,
                -0.0195327484,
                -0.0054103808,
                0.0347564518,
                0.5120802482,
                -0.0329902864
            ),
            doubleArrayOf(
                -0.5933332356,
                -0.2488721082,
                0.2357173629,
                0.0177285473,
                0.0856630593,
                -0.3567126300,
                -0.1600668126,
                -0.1010899621
            ),
            doubleArrayOf(
                -0.0514349819,
                -0.0854319435,
                0.1125050061,
                0.0063453560,
                -0.2250000688,
                -0.2209343090,
                0.1964623477,
                -0.1512329924
            ),
            doubleArrayOf(
                0.0197395947,
                -0.1997170581,
                -0.1425959019,
                -0.2749477910,
                -0.0969467073,
                0.0603688520,
                -0.2826905192,
                0.1794315473
            )
        )
        val m: RealMatrix = data.toNDArray()
        val s: RealMatrix = checkAEqualPTPt(m)
        checkSchurForm(s)
    }

    ///////////////////////////////////////////////////////////////////////////
    // Test helpers
    ///////////////////////////////////////////////////////////////////////////
    private fun checkAEqualPTPt(matrix: RealMatrix): RealMatrix {
        val transformer = SchurTransformer(matrix)
        val p: RealMatrix = transformer.getP()
        val t: RealMatrix = transformer.getT()
        val pT: RealMatrix = transformer.getPT()
        val result: RealMatrix = mk.linalg.dot(mk.linalg.dot(p, t), pT)
        val norm: Double = (result - matrix).getNorm()
        assertEquals(0.0, norm, 1.0e-9)
        return t
    }

    private fun checkOrthogonal(m: RealMatrix) {
        val mTm: RealMatrix = mk.linalg.dot(m.transpose(), m)
        val id: RealMatrix = mk.identity(mTm.getRowDimension())
        assertEquals(0.0, (mTm - id).getNorm(), 1.0e-14)
    }

    private fun checkSchurForm(m: RealMatrix) {
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

}