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

/** This class implements the WELL19937c pseudo-random number generator
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
class Well19937c
/** Creates a new random number generator.
 *
 * The instance is initialized using the current time as the
 * seed.
 */
    : AbstractWell(K, M1, M2, M3) {

    /** {@inheritDoc}  */
    override fun next(bits: Int): Int {
        val indexRm1: Int = iRm1[index]
        val indexRm2: Int = iRm2[index]
        val v0: Int = v[index]
        val vM1: Int = v[i1[index]]
        val vM2: Int = v[i2[index]]
        val vM3: Int = v[i3[index]]
        val z0 = -0x80000000 and v[indexRm1] xor (0x7FFFFFFF and v.get(indexRm2))
        val z1 = v0 xor (v0 shl 25) xor (vM1 xor (vM1 ushr 27))
        val z2 = vM2 ushr 9 xor (vM3 xor (vM3 ushr 1))
        val z3 = z1 xor z2
        var z4 = z0 xor (z1 xor (z1 shl 9)) xor (z2 xor (z2 shl 21)) xor (z3 xor (z3 ushr 21))
        v[index] = z3
        v[indexRm1] = z4
        v[indexRm2] = v[indexRm2] and -0x80000000
        index = indexRm1


        // add Matsumoto-Kurita tempering
        // to get a maximally-equidistributed generator
        z4 = z4 xor (z4 shl 7 and -0x1b91e900)
        z4 = z4 xor (z4 shl 15 and -0x64798000)
        return z4 ushr 32 - bits
    }

    companion object {
        /** Serializable version identifier.  */
        private const val serialVersionUID = -7203498180754925124L

        /** Number of bits in the pool.  */
        private const val K = 19937

        /** First parameter of the algorithm.  */
        private const val M1 = 70

        /** Second parameter of the algorithm.  */
        private const val M2 = 179

        /** Third parameter of the algorithm.  */
        private const val M3 = 449
    }
}