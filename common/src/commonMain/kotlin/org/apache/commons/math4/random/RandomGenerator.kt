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

/**
 * Interface extracted from `java.util.Random`.
 *
 * @since 1.1
 */
interface RandomGenerator {
    /**
     * Sets the seed of the underlying random number generator using an
     * `int` seed.
     *
     * Sequences of values generated starting with the same seeds
     * should be identical.
     *
     * @param seed the seed value
     */
    fun setSeed(seed: Int)

    /**
     * Sets the seed of the underlying random number generator using an
     * `int` array seed.
     *
     * Sequences of values generated starting with the same seeds
     * should be identical.
     *
     * @param seed the seed value
     */
    fun setSeed(seed: IntArray?)

    /**
     * Sets the seed of the underlying random number generator using a
     * `long` seed.
     *
     * Sequences of values generated starting with the same seeds
     * should be identical.
     *
     * @param seed the seed value
     */
    fun setSeed(seed: Long)

    /**
     * Generates random bytes and places them into a user-supplied
     * byte array.  The number of random bytes produced is equal to
     * the length of the byte array.
     *
     * @param bytes the non-null byte array in which to put the
     * random bytes
     */
    fun nextBytes(bytes: ByteArray?)

    /**
     * Returns the next pseudorandom, uniformly distributed `int`
     * value from this random number generator's sequence.
     * All 2<sup style="font-size: smaller">32</sup> possible `int` values
     * should be produced with  (approximately) equal probability.
     *
     * @return the next pseudorandom, uniformly distributed `int`
     * value from this random number generator's sequence
     */
    fun nextInt(): Int

    /**
     * Returns a pseudorandom, uniformly distributed `int` value
     * between 0 (inclusive) and the specified value (exclusive), drawn from
     * this random number generator's sequence.
     *
     * @param n the bound on the random number to be returned.  Must be
     * positive.
     * @return  a pseudorandom, uniformly distributed `int`
     * value between 0 (inclusive) and n (exclusive).
     *
     */
    fun nextInt(n: Int): Int

    /**
     * Returns the next pseudorandom, uniformly distributed `long`
     * value from this random number generator's sequence.  All
     * 2<sup style="font-size: smaller">64</sup> possible `long` values
     * should be produced with (approximately) equal probability.
     *
     * @return  the next pseudorandom, uniformly distributed `long`
     * value from this random number generator's sequence
     */
    fun nextLong(): Long

    /**
     * Returns the next pseudorandom, uniformly distributed
     * `boolean` value from this random number generator's
     * sequence.
     *
     * @return  the next pseudorandom, uniformly distributed
     * `boolean` value from this random number generator's
     * sequence
     */
    fun nextBoolean(): Boolean

    /**
     * Returns the next pseudorandom, uniformly distributed `float`
     * value between `0.0` and `1.0` from this random
     * number generator's sequence.
     *
     * @return  the next pseudorandom, uniformly distributed `float`
     * value between `0.0` and `1.0` from this
     * random number generator's sequence
     */
    fun nextFloat(): Float

    /**
     * Returns the next pseudorandom, uniformly distributed
     * `double` value between `0.0` and
     * `1.0` from this random number generator's sequence.
     *
     * @return  the next pseudorandom, uniformly distributed
     * `double` value between `0.0` and
     * `1.0` from this random number generator's sequence
     */
    fun nextDouble(): Double

    /**
     * Returns the next pseudorandom, Gaussian ("normally") distributed
     * `double` value with mean `0.0` and standard
     * deviation `1.0` from this random number generator's sequence.
     *
     * @return  the next pseudorandom, Gaussian ("normally") distributed
     * `double` value with mean `0.0` and
     * standard deviation `1.0` from this random number
     * generator's sequence
     */
    fun nextGaussian(): Double
}