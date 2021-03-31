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
package org.apache.commons.math4.random

import kotlin.math.*

/** Base class for random number generators that generates bits streams.
 *
 * @since 2.0
 */
abstract class BitsStreamGenerator : RandomGenerator {
    /** Next gaussian.  */
    private var nextGaussian: Double

    /** {@inheritDoc}  */
    abstract override fun setSeed(seed: Int)

    /** {@inheritDoc}  */
    abstract override fun setSeed(seed: IntArray?)

    /** {@inheritDoc}  */
    abstract override fun setSeed(seed: Long)

    /** Generate next pseudorandom number.
     *
     * This method is the core generation algorithm. It is used by all the
     * public generation methods for the various primitive types [ ][.nextBoolean], [.nextBytes], [.nextDouble],
     * [.nextFloat], [.nextGaussian], [.nextInt],
     * [.next] and [.nextLong].
     * @param bits number of random bits to produce
     * @return random bits generated
     */
    protected abstract fun next(bits: Int): Int

    /** {@inheritDoc}  */
    override fun nextBoolean(): Boolean {
        return next(1) != 0
    }

    /** {@inheritDoc}  */
    override fun nextBytes(bytes: ByteArray?) {
        var i = 0
        val iEnd = bytes!!.size - 3
        while (i < iEnd) {
            val random = next(32)
            bytes[i] = (random and 0xff).toByte()
            bytes[i + 1] = (random shr 8 and 0xff).toByte()
            bytes[i + 2] = (random shr 16 and 0xff).toByte()
            bytes[i + 3] = (random shr 24 and 0xff).toByte()
            i += 4
        }
        var random = next(32)
        while (i < bytes.size) {
            bytes[i++] = (random and 0xff).toByte()
            random = random shr 8
        }
    }

    /** {@inheritDoc}  */
    override fun nextDouble(): Double {
        val high = next(26).toLong() shl 26
        val low = next(26).toLong()
        return (high or low).toDouble() * 2.220446049250313e-16
    }

    /** {@inheritDoc}  */
    override fun nextFloat(): Float {
        return next(23) * 1.1920929E-7f
    }

    /** {@inheritDoc}  */
    override fun nextGaussian(): Double {
        val random: Double
        if (nextGaussian.isNaN()) {
            // generate a new pair of gaussian numbers
            val x = nextDouble()
            val y = nextDouble()
            val alpha: Double = 2 * PI * x
            val r: Double = sqrt(
                -2 * ln(y)
            )
            random = r * cos(alpha)
            nextGaussian = r * sin(alpha)
        } else {
            // use the second element of the pair already generated
            random = nextGaussian
            nextGaussian = Double.NaN
        }
        return random
    }

    /** {@inheritDoc}  */
    override fun nextInt(): Int {
        return next(32)
    }

    /**
     * {@inheritDoc}
     *
     * This default implementation is copied from Apache Harmony
     * java.util.Random (r929253).
     *
     *
     * Implementation notes:
     *  * If n is a power of 2, this method returns
     * `(int) ((n * (long) next(31)) >> 31)`.
     *
     *  * If n is not a power of 2, what is returned is `next(31) % n`
     * with `next(31)` values rejected (i.e. regenerated) until a
     * value that is larger than the remainder of `Integer.MAX_VALUE / n`
     * is generated. Rejection of this initial segment is necessary to ensure
     * a uniform distribution.
     */
    override fun nextInt(n: Int): Int {
        if (n > 0) {
            if (n and -n == n) {
                return (n * next(31).toLong() shr 31).toInt()
            }
            var bits: Int
            var `val`: Int
            do {
                bits = next(31)
                `val` = bits % n
            } while (bits - `val` + (n - 1) < 0)
            return `val`
        } else throw IllegalArgumentException("Not Strictly Positive: $n")
    }

    /** {@inheritDoc}  */
    override fun nextLong(): Long {
        val high = next(32).toLong() shl 32
        val low = next(32).toLong() and 0xffffffffL
        return high or low
    }

    /**
     * Returns a pseudorandom, uniformly distributed `long` value
     * between 0 (inclusive) and the specified value (exclusive), drawn from
     * this random number generator's sequence.
     *
     * @param n the bound on the random number to be returned.  Must be
     * positive.
     * @return  a pseudorandom, uniformly distributed `long`
     * value between 0 (inclusive) and n (exclusive).
     * @throws IllegalArgumentException  if n is not positive.
     */
    fun nextLong(n: Long): Long {
        if (n > 0) {
            var bits: Long
            var `val`: Long
            do {
                bits = next(31).toLong() shl 32
                bits = bits or (next(32).toLong() and 0xffffffffL)
                `val` = bits % n
            } while (bits - `val` + (n - 1) < 0)
            return `val`
        } else throw IllegalArgumentException("Not Strictly Positive: $n")
    }

    /**
     * Clears the cache used by the default implementation of
     * [.nextGaussian].
     */
    fun clear() {
        nextGaussian = Double.NaN
    }

    /**
     * Creates a new random number generator.
     */
    init {
        nextGaussian = Double.NaN
    }

}