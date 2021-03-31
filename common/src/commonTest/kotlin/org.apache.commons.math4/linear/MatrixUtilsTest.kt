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
import org.jetbrains.kotlinx.multik.ndarray.data.RealMatrix
import org.jetbrains.kotlinx.multik.ndarray.data.toNDArray
import kotlin.math.ulp
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlin.test.fail


/**
 * Test cases for the [MatrixUtils] class.
 *
 */
class MatrixUtilsTest {
    var testData: Array<DoubleArray> = arrayOf(
        doubleArrayOf(1.0, 2.0, 3.0),
        doubleArrayOf(2.0, 5.0, 3.0),
        doubleArrayOf(1.0, 0.0, 8.0)
    )
    var testData3x3Singular: Array<DoubleArray> = arrayOf(
        doubleArrayOf(1.0, 4.0, 7.0),
        doubleArrayOf(2.0, 5.0, 8.0),
        doubleArrayOf(3.0, 6.0, 9.0)
    )
    var testData3x4: Array<DoubleArray> = arrayOf(
        doubleArrayOf(12.0, -51.0, 4.0, 1.0),
        doubleArrayOf(6.0, 167.0, -68.0, 2.0),
        doubleArrayOf(-4.0, 24.0, -41.0, 3.0)
    )
    var row: DoubleArray = doubleArrayOf(1.0, 2.0, 3.0)
    var col: DoubleArray = doubleArrayOf(0.0, 4.0, 6.0)

    @Test
    fun testIsSymmetric() {
        val eps = 1.0.ulp
        val dataSym = arrayOf(
            doubleArrayOf(1.0, 2.0, 3.0),
            doubleArrayOf(2.0, 2.0, 5.0),
            doubleArrayOf(3.0, 5.0, 6.0)
        )
        assertTrue(MatrixUtils.isSymmetric(dataSym.toNDArray(), eps))
        val dataNonSym = arrayOf(
            doubleArrayOf(1.0, 2.0, -3.0),
            doubleArrayOf(2.0, 2.0, 5.0),
            doubleArrayOf(3.0, 5.0, 6.0)
        )
        assertFalse(MatrixUtils.isSymmetric(dataNonSym.toNDArray(), eps))
    }

    @Test
    fun testIsSymmetricTolerance() {
        val eps = 1e-4
        val dataSym1 = arrayOf(
            doubleArrayOf(1.0, 1.0, 1.00009),
            doubleArrayOf(1.0, 1.0, 1.0),
            doubleArrayOf(1.0, 1.0, 1.0)
        )
        assertTrue(MatrixUtils.isSymmetric(dataSym1.toNDArray(), eps))
        val dataSym2 = arrayOf(
            doubleArrayOf(1.0, 1.0, 0.99990),
            doubleArrayOf(1.0, 1.0, 1.0),
            doubleArrayOf(1.0, 1.0, 1.0)
        )
        assertTrue(MatrixUtils.isSymmetric(dataSym2.toNDArray(), eps))
        val dataNonSym1 = arrayOf(
            doubleArrayOf(1.0, 1.0, 1.00011),
            doubleArrayOf(1.0, 1.0, 1.0),
            doubleArrayOf(1.0, 1.0, 1.0)
        )
        assertFalse(MatrixUtils.isSymmetric(dataNonSym1.toNDArray(), eps))
        val dataNonSym2 = arrayOf(
            doubleArrayOf(1.0, 1.0, 0.99989),
            doubleArrayOf(1.0, 1.0, 1.0),
            doubleArrayOf(1.0, 1.0, 1.0)
        )
        assertFalse(MatrixUtils.isSymmetric(dataNonSym2.toNDArray(), eps))
    }

    @Test
    fun testInverseSingular() {
        val m: RealMatrix = testData3x3Singular.toNDArray()
        try {
            MatrixUtils.inverse(m)
            fail("Expected SingularMatrixException")
        } catch (t: Throwable) {
            // Expected SingularMatrixException
        }

    }

    @Test
    fun testInverseNonSquare() {
        val m: RealMatrix = testData3x4.toNDArray()
        try {
            MatrixUtils.inverse(m)
            fail("Expected NonSquareMatrixException")
        } catch (t: Throwable) {
            // Expected NonSquareMatrixException
        }
    }

    @Test
    fun testInverseRealMatrix() {
        val m: RealMatrix = testData.toNDArray()
        val inverse: RealMatrix = MatrixUtils.inverse(m)
        val result: RealMatrix = mk.linalg.dot(m, inverse)
        assertEquals(
            "MatrixUtils.inverse() returns wrong result",
            mk.identity(testData.size),
            result,
            1e-12
        )
    }

}