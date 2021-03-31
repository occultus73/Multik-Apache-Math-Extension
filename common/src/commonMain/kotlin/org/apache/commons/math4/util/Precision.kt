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
package org.apache.commons.math4.util

import kotlin.math.abs
import kotlin.math.max

/**
 * Utilities for comparing numbers.
 *
 * @since 3.0
 */
object Precision {
    /**
     *
     *
     * Largest double-precision floating-point number such that
     * `1 + EPSILON` is numerically equal to 1. This value is an upper
     * bound on the relative error due to rounding real numbers to double
     * precision floating-point numbers.
     *
     *
     *
     * In IEEE 754 arithmetic, this is 2<sup>-53</sup>.
     *
     *
     * @see [Machine epsilon](http://en.wikipedia.org/wiki/Machine_epsilon)
     */
    val EPSILON: Double

    /**
     * Safe minimum, such that `1 / SAFE_MIN` does not overflow.
     * <br></br>
     * In IEEE 754 arithmetic, this is also the smallest normalized
     * number 2<sup>-1022</sup>.
     */
    val SAFE_MIN: Double

    /** Exponent offset in IEEE754 representation.  */
    private const val EXPONENT_OFFSET = 1023L

    /** Offset to order signed double numbers lexicographically.  */
    private const val SGN_MASK = (-0x800000000000000L)

    /** Offset to order signed double numbers lexicographically.  */
    private const val SGN_MASK_FLOAT = -0x80000000

    /** Positive zero.  */
    private const val POSITIVE_ZERO = 0.0

    /** Positive zero bits.  */
    private val POSITIVE_ZERO_DOUBLE_BITS = (+0.0).toRawBits()

    /** Negative zero bits.  */
    private val NEGATIVE_ZERO_DOUBLE_BITS = (-0.0).toRawBits()

    /** Positive zero bits.  */
    private val POSITIVE_ZERO_FLOAT_BITS = (+0.0f).toRawBits()

    /** Negative zero bits.  */
    private val NEGATIVE_ZERO_FLOAT_BITS = (-0.0f).toRawBits()

    /**
     * Compares two numbers given some amount of allowed error.
     *
     * @param x the first number
     * @param y the second number
     * @param eps the amount of error to allow when checking for equality
     * @return  * 0 if  [equals(x, y, eps)][.equals]
     *  * &lt; 0 if ![equals(x, y, eps)][.equals] &amp;&amp; x &lt; y
     *  * > 0 if ![equals(x, y, eps)][.equals] &amp;&amp; x > y
     */
    fun compareTo(x: Double, y: Double, eps: Double): Int {
        if (equals(x, y, eps)) {
            return 0
        } else if (x < y) {
            return -1
        }
        return 1
    }

    /**
     * Compares two numbers given some amount of allowed error.
     * Two float numbers are considered equal if there are `(maxUlps - 1)`
     * (or fewer) floating point numbers between them, i.e. two adjacent floating
     * point numbers are considered equal.
     * Adapted from [
 * Bruce Dawson](http://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/)
     *
     * @param x first value
     * @param y second value
     * @param maxUlps `(maxUlps - 1)` is the number of floating point
     * values between `x` and `y`.
     * @return  * 0 if  [equals(x, y, maxUlps)][.equals]
     *  * &lt; 0 if ![equals(x, y, maxUlps)][.equals] &amp;&amp; x &lt; y
     *  * > 0 if ![equals(x, y, maxUlps)][.equals] &amp;&amp; x > y
     */
    fun compareTo(x: Double, y: Double, maxUlps: Int): Int {
        if (equals(x, y, maxUlps)) {
            return 0
        } else if (x < y) {
            return -1
        }
        return 1
    }

    /**
     * Returns true if both arguments are NaN or neither is NaN and they are
     * equal as defined by [equals(x, y, 1)][.equals].
     *
     * @param x first value
     * @param y second value
     * @return `true` if the values are equal or both are NaN.
     * @since 2.2
     */
    fun equalsIncludingNaN(x: Float, y: Float): Boolean {
        return if (x != x || y != y) (x != x) xor (y == y) else equals(x, y, 1)
    }

    /**
     * Returns true if both arguments are equal or within the range of allowed
     * error (inclusive).
     *
     * @param x first value
     * @param y second value
     * @param eps the amount of absolute error to allow.
     * @return `true` if the values are equal or within range of each other.
     * @since 2.2
     */
    fun equals(x: Float, y: Float, eps: Float): Boolean {
        return equals(x, y, 1) || abs(y - x) <= eps
    }

    /**
     * Returns true if both arguments are NaN or are equal or within the range
     * of allowed error (inclusive).
     *
     * @param x first value
     * @param y second value
     * @param eps the amount of absolute error to allow.
     * @return `true` if the values are equal or within range of each other,
     * or both are NaN.
     * @since 2.2
     */
    fun equalsIncludingNaN(x: Float, y: Float, eps: Float): Boolean {
        return equalsIncludingNaN(x, y) || abs(y - x) <= eps
    }
    /**
     * Returns true if both arguments are equal or within the range of allowed
     * error (inclusive).
     * Two float numbers are considered equal if there are `(maxUlps - 1)`
     * (or fewer) floating point numbers between them, i.e. two adjacent floating
     * point numbers are considered equal.
     * Adapted from [
 * Bruce Dawson](http://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/)
     *
     * @param x first value
     * @param y second value
     * @param maxUlps `(maxUlps - 1)` is the number of floating point
     * values between `x` and `y`.
     * @return `true` if there are fewer than `maxUlps` floating
     * point values between `x` and `y`.
     * @since 2.2
     */
    /**
     * Returns true iff they are equal as defined by
     * [equals(x, y, 1)][.equals].
     *
     * @param x first value
     * @param y second value
     * @return `true` if the values are equal.
     */
    fun equals(x: Float, y: Float, maxUlps: Int = 1): Boolean {
        val xInt = x.toRawBits()
        val yInt = y.toRawBits()
        val isEqual: Boolean
        if (xInt xor yInt and SGN_MASK_FLOAT == 0) {
            // number have same sign, there is no risk of overflow
            isEqual = abs(xInt - yInt) <= maxUlps
        } else {
            // number have opposite signs, take care of overflow
            val deltaPlus: Int
            val deltaMinus: Int
            if (xInt < yInt) {
                deltaPlus = yInt - POSITIVE_ZERO_FLOAT_BITS
                deltaMinus = xInt - NEGATIVE_ZERO_FLOAT_BITS
            } else {
                deltaPlus = xInt - POSITIVE_ZERO_FLOAT_BITS
                deltaMinus = yInt - NEGATIVE_ZERO_FLOAT_BITS
            }
            isEqual = if (deltaPlus > maxUlps) {
                false
            } else {
                deltaMinus <= maxUlps - deltaPlus
            }
        }
        return isEqual && !x.isNaN() && !y.isNaN()
    }

    /**
     * Returns true if both arguments are NaN or if they are equal as defined
     * by [equals(x, y, maxUlps)][.equals].
     *
     * @param x first value
     * @param y second value
     * @param maxUlps `(maxUlps - 1)` is the number of floating point
     * values between `x` and `y`.
     * @return `true` if both arguments are NaN or if there are less than
     * `maxUlps` floating point values between `x` and `y`.
     * @since 2.2
     */
    fun equalsIncludingNaN(x: Float, y: Float, maxUlps: Int): Boolean {
        return if (x != x || y != y) (x != x) xor (y == y) else equals(x, y, maxUlps)
    }

    /**
     * Returns true if both arguments are NaN or neither is NaN and they are
     * equal as defined by [equals(x, y, 1)][.equals].
     *
     * @param x first value
     * @param y second value
     * @return `true` if the values are equal or both are NaN.
     * @since 2.2
     */
    fun equalsIncludingNaN(x: Double, y: Double): Boolean {
        return if (x != x || y != y) (x != x) xor (y == y) else equals(x, y, 1)
    }

    /**
     * Returns `true` if there is no double value strictly between the
     * arguments or the difference between them is within the range of allowed
     * error (inclusive).
     *
     * @param x First value.
     * @param y Second value.
     * @param eps Amount of allowed absolute error.
     * @return `true` if the values are two adjacent floating point
     * numbers or they are within range of each other.
     */
    fun equals(x: Double, y: Double, eps: Double): Boolean {
        return equals(x, y, 1) || abs(y - x) <= eps
    }

    /**
     * Returns `true` if there is no double value strictly between the
     * arguments or the relative difference between them is smaller or equal
     * to the given tolerance.
     *
     * @param x First value.
     * @param y Second value.
     * @param eps Amount of allowed relative error.
     * @return `true` if the values are two adjacent floating point
     * numbers or they are within range of each other.
     * @since 3.1
     */
    fun equalsWithRelativeTolerance(x: Double, y: Double, eps: Double): Boolean {
        if (equals(x, y, 1)) {
            return true
        }
        val absoluteMax = max(abs(x), abs(y))
        val relativeDifference = abs((x - y) / absoluteMax)
        return relativeDifference <= eps
    }

    /**
     * Returns true if both arguments are NaN or are equal or within the range
     * of allowed error (inclusive).
     *
     * @param x first value
     * @param y second value
     * @param eps the amount of absolute error to allow.
     * @return `true` if the values are equal or within range of each other,
     * or both are NaN.
     * @since 2.2
     */
    fun equalsIncludingNaN(x: Double, y: Double, eps: Double): Boolean {
        return equalsIncludingNaN(x, y) || abs(y - x) <= eps
    }
    /**
     * Returns true if both arguments are equal or within the range of allowed
     * error (inclusive).
     *
     * Two float numbers are considered equal if there are `(maxUlps - 1)`
     * (or fewer) floating point numbers between them, i.e. two adjacent
     * floating point numbers are considered equal.
     *
     * Adapted from [Bruce Dawson](http://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/)
     *
     * @param x first value
     * @param y second value
     * @param maxUlps `(maxUlps - 1)` is the number of floating point
     * values between `x` and `y`.
     * @return `true` if there are fewer than `maxUlps` floating
     * point values between `x` and `y`.
     */
    /**
     * Returns true iff they are equal as defined by
     * [equals(x, y, 1)][.equals].
     *
     * @param x first value
     * @param y second value
     * @return `true` if the values are equal.
     */
    fun equals(x: Double, y: Double, maxUlps: Int = 1): Boolean {
        val xInt = x.toRawBits()
        val yInt = y.toRawBits()
        val isEqual: Boolean
        if (xInt xor yInt and SGN_MASK == 0L) {
            // number have same sign, there is no risk of overflow
            isEqual = abs(xInt - yInt) <= maxUlps
        } else {
            // number have opposite signs, take care of overflow
            val deltaPlus: Long
            val deltaMinus: Long
            if (xInt < yInt) {
                deltaPlus = yInt - POSITIVE_ZERO_DOUBLE_BITS
                deltaMinus = xInt - NEGATIVE_ZERO_DOUBLE_BITS
            } else {
                deltaPlus = xInt - POSITIVE_ZERO_DOUBLE_BITS
                deltaMinus = yInt - NEGATIVE_ZERO_DOUBLE_BITS
            }
            isEqual = if (deltaPlus > maxUlps) {
                false
            } else {
                deltaMinus <= maxUlps - deltaPlus
            }
        }
        return isEqual && !(x.isNaN()) && !(y.isNaN())
    }

    /**
     * Returns true if both arguments are NaN or if they are equal as defined
     * by [equals(x, y, maxUlps)][.equals].
     *
     * @param x first value
     * @param y second value
     * @param maxUlps `(maxUlps - 1)` is the number of floating point
     * values between `x` and `y`.
     * @return `true` if both arguments are NaN or if there are less than
     * `maxUlps` floating point values between `x` and `y`.
     * @since 2.2
     */
    fun equalsIncludingNaN(x: Double, y: Double, maxUlps: Int): Boolean {
        return if (x != x || y != y) (x != x) xor (y == y) else equals(x, y, maxUlps)
    }

    /**
     * Computes a number `delta` close to `originalDelta` with
     * the property that <pre>`
     * x + delta - x
    `</pre> *
     * is exactly machine-representable.
     * This is useful when computing numerical derivatives, in order to reduce
     * roundoff errors.
     *
     * @param x Value.
     * @param originalDelta Offset value.
     * @return a number `delta` so that `x + delta` and `x`
     * differ by a representable floating number.
     */
    fun representableDelta(
        x: Double,
        originalDelta: Double
    ): Double {
        return x + originalDelta - x
    }

    init {
        /*
         *  This was previously expressed as = 0x1.0p-53;
         *  However, OpenJDK (Sparc Solaris) cannot handle such small
         *  constants: MATH-721
         */
        EPSILON = Double.fromBits(EXPONENT_OFFSET - 53L shl 52)

        /*
         * This was previously expressed as = 0x1.0p-1022;
         * However, OpenJDK (Sparc Solaris) cannot handle such small
         * constants: MATH-721
         */
        SAFE_MIN = Double.fromBits(EXPONENT_OFFSET - 1022L shl 52)
    }

}