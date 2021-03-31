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
package org.apache.commons.math4

import org.jetbrains.kotlinx.multik.ndarray.data.RealMatrix
import org.jetbrains.kotlinx.multik.ndarray.data.getColumnDimension
import org.jetbrains.kotlinx.multik.ndarray.data.getNorm
import org.jetbrains.kotlinx.multik.ndarray.data.getRowDimension
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import kotlin.math.abs
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 */
object TestUtils {

    fun <T> assertEquals(expected: T, actual: T) = kotlin.test.assertEquals(expected, actual)

    /**
     * Verifies that expected and actual are within delta, or are both NaN or
     * infinities of the same sign.
     */
    fun assertEquals(expected: Double, actual: Double, delta: Double) {
        assertEquals(null, expected, actual, delta)
    }

    /**
     * Verifies that expected and actual are within delta, or are both NaN or
     * infinities of the same sign.
     */
    fun assertEquals(msg: String?, expected: Double, actual: Double, delta: Double) {
        // check for NaN
        if (expected.isNaN()) {
            assertTrue(actual.isNaN(), "$actual is not NaN.")
        } else {
            assertTrue(
                msg ?: "Expected: $expected, got: $actual"
            ) { abs(expected - actual) <= delta }
        }
    }

    /** verifies that two matrices are close (1-norm)  */
    fun assertEquals(msg: String, expected: RealMatrix, observed: RealMatrix, tolerance: Double) {
        assertNotNull(observed, "$msg\nObserved should not be null")
        if (expected.getColumnDimension() != observed.getColumnDimension() || expected.getRowDimension() != observed.getRowDimension()) {
            val messageBuffer = StringBuilder(msg)
            messageBuffer.append("\nObserved has incorrect dimensions.")
            messageBuffer.append("observed is ${observed.getRowDimension()} x ${observed.getColumnDimension()}")
            messageBuffer.append("expected ${expected.getRowDimension()} x ${expected.getColumnDimension()}")
            fail(messageBuffer.toString())
        }
        val delta: RealMatrix = expected - observed
        if (delta.getNorm() >= tolerance) {
            val messageBuffer = StringBuilder(msg)
            messageBuffer.append("\nExpected: $expected")
            messageBuffer.append("\nObserved: $observed")
            messageBuffer.append("\nexpected - observed: $delta")
            fail(messageBuffer.toString())
        }
    }

}