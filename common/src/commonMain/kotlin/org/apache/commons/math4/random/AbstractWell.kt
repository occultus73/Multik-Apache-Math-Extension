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

import kotlin.math.min
import kotlin.random.Random

/** This abstract class implements the WELL class of pseudo-random number generator
 * from Franois Panneton, Pierre L'Ecuyer and Makoto Matsumoto.
 *
 *
 * This generator is described in a paper by Franois Panneton,
 * Pierre L'Ecuyer and Makoto Matsumoto [Improved
 * Long-Period Generators Based on Linear Recurrences Modulo 2](http://www.iro.umontreal.ca/~lecuyer/myftp/papers/wellrng.pdf) ACM
 * Transactions on Mathematical Software, 32, 1 (2006). The errata for the paper
 * are in [wellrng-errata.txt](http://www.iro.umontreal.ca/~lecuyer/myftp/papers/wellrng-errata.txt).
 *
 * @see [WELL Random number generator](http://www.iro.umontreal.ca/~panneton/WELLRNG.html)
 *
 * @since 2.2
 */
abstract class AbstractWell protected constructor(
    k: Int,
    m1: Int,
    m2: Int,
    m3: Int,
    seed: IntArray? = null
) : BitsStreamGenerator() {
    /** Current index in the bytes pool.  */
    protected var index: Int

    /** Bytes pool.  */
    protected val v: IntArray

    /** Index indirection table giving for each index its predecessor taking table size into account.  */
    protected val iRm1: IntArray

    /** Index indirection table giving for each index its second predecessor taking table size into account.  */
    protected val iRm2: IntArray

    /** Index indirection table giving for each index the value index + m1 taking table size into account.  */
    protected val i1: IntArray

    /** Index indirection table giving for each index the value index + m2 taking table size into account.  */
    protected val i2: IntArray

    /** Index indirection table giving for each index the value index + m3 taking table size into account.  */
    protected val i3: IntArray

    /** Reinitialize the generator as if just built with the given int seed.
     *
     * The state of the generator is exactly the same as a new
     * generator built with the same seed.
     * @param seed the initial seed (32 bits integer)
     */
    override fun setSeed(seed: Int) {
        setSeed(intArrayOf(seed))
    }

    /** Reinitialize the generator as if just built with the given int array seed.
     *
     * The state of the generator is exactly the same as a new
     * generator built with the same seed.
     * @param seed the initial seed (32 bits integers array). If null
     * the seed of the generator will be the system time plus the system identity
     * hash code of the instance.
     */
    final override fun setSeed(seed: IntArray?) {
        if (seed == null) {
            setSeed(Random.nextInt().toLong() + hashCode())
            return
        }
        seed.copyInto(v, endIndex = min(seed.size, v.size))
        if (seed.size < v.size) {
            for (i in seed.size until v.size) {
                val l = v[i - seed.size].toLong()
                v[i] = (1812433253L * (l xor (l shr 30)) + i and 0xffffffffL).toInt()
            }
        }
        index = 0
        clear() // Clear normal deviate cache
    }

    /** Reinitialize the generator as if just built with the given long seed.
     *
     * The state of the generator is exactly the same as a new
     * generator built with the same seed.
     * @param seed the initial seed (64 bits integer)
     */
    override fun setSeed(seed: Long) {
        setSeed(intArrayOf((seed ushr 32).toInt(), (seed and 0xffffffffL).toInt()))
        Random(65)
    }

    /** Creates a new random number generator using an int array seed.
     * @param k number of bits in the pool (not necessarily a multiple of 32)
     * @param m1 first parameter of the algorithm
     * @param m2 second parameter of the algorithm
     * @param m3 third parameter of the algorithm
     * @param seed the initial seed (32 bits integers array), if null
     * the seed of the generator will be related to the current time
     */
    init {

        // the bits pool contains k bits, k = r w - p where r is the number
        // of w bits blocks, w is the block size (always 32 in the original paper)
        // and p is the number of unused bits in the last block
        val w = 32
        val r = (k + w - 1) / w
        v = IntArray(r)
        index = 0

        // precompute indirection index tables. These tables are used for optimizing access
        // they allow saving computations like "(j + r - 2) % r" with costly modulo operations
        iRm1 = IntArray(r)
        iRm2 = IntArray(r)
        i1 = IntArray(r)
        i2 = IntArray(r)
        i3 = IntArray(r)
        for (j in 0 until r) {
            iRm1[j] = (j + r - 1) % r
            iRm2[j] = (j + r - 2) % r
            i1[j] = (j + m1) % r
            i2[j] = (j + m2) % r
            i3[j] = (j + m3) % r
        }

        // initialize the pool content
        setSeed(seed)
    }
}