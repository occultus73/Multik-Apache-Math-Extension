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
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.math.ulp
import kotlin.random.Random
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.fail

class EigenDecompositionTest {
    lateinit var refValues: DoubleArray
    lateinit var matrix: RealMatrix

    @Test
    fun testDimension1() {
        val matrix: RealMatrix = arrayOf(doubleArrayOf(1.5)).toNDArray()
        val ed: EigenDecomposition?
        ed = EigenDecomposition(matrix)
        assertEquals(1.5, ed.getRealEigenvalue(0), 1.0e-15)
    }

    @Test
    fun testDimension2() {
        val matrix: RealMatrix =
            arrayOf(doubleArrayOf(59.0, 12.0), doubleArrayOf(12.0, 66.0)).toNDArray()
        val ed = EigenDecomposition(matrix)
        assertEquals(75.0, ed.getRealEigenvalue(0), 1.0e-15)
        assertEquals(50.0, ed.getRealEigenvalue(1), 1.0e-15)
    }

    @Test
    fun testDimension3() {
        val matrix: RealMatrix = arrayOf(
            doubleArrayOf(39632.0, -4824.0, -16560.0),
            doubleArrayOf(-4824.0, 8693.0, 7920.0),
            doubleArrayOf(-16560.0, 7920.0, 17300.0)
        ).toNDArray()
        val ed = EigenDecomposition(matrix)
        assertEquals(50000.0, ed.getRealEigenvalue(0), 3.0e-11)
        assertEquals(12500.0, ed.getRealEigenvalue(1), 3.0e-11)
        assertEquals(3125.0, ed.getRealEigenvalue(2), 3.0e-11)
    }

    @Test
    fun testDimension3MultipleRoot() {
        val matrix: RealMatrix = arrayOf(
            doubleArrayOf(5.0, 10.0, 15.0),
            doubleArrayOf(10.0, 20.0, 30.0),
            doubleArrayOf(15.0, 30.0, 45.0)
        ).toNDArray()
        val ed = EigenDecomposition(matrix)
        assertEquals(70.0, ed.getRealEigenvalue(0), 3.0e-11)
        assertEquals(0.0, ed.getRealEigenvalue(1), 3.0e-11)
        assertEquals(0.0, ed.getRealEigenvalue(2), 3.0e-11)
    }

    @Test
    fun testDimension4WithSplit() {
        val matrix: RealMatrix = arrayOf(
            doubleArrayOf(0.784, -0.288, 0.000, 0.000),
            doubleArrayOf(-0.288, 0.616, 0.000, 0.000),
            doubleArrayOf(0.000, 0.000, 0.164, -0.048),
            doubleArrayOf(0.000, 0.000, -0.048, 0.136)
        ).toNDArray()
        val ed = EigenDecomposition(matrix)
        assertEquals(1.0, ed.getRealEigenvalue(0), 1.0e-15)
        assertEquals(0.4, ed.getRealEigenvalue(1), 1.0e-15)
        assertEquals(0.2, ed.getRealEigenvalue(2), 1.0e-15)
        assertEquals(0.1, ed.getRealEigenvalue(3), 1.0e-15)
    }

    @Test
    fun testDimension4WithoutSplit() {
        val matrix: RealMatrix = arrayOf(
            doubleArrayOf(0.5608, -0.2016, 0.1152, -0.2976),
            doubleArrayOf(-0.2016, 0.4432, -0.2304, 0.1152),
            doubleArrayOf(0.1152, -0.2304, 0.3088, -0.1344),
            doubleArrayOf(-0.2976, 0.1152, -0.1344, 0.3872)
        ).toNDArray()
        val ed = EigenDecomposition(matrix)
        assertEquals(1.0, ed.getRealEigenvalue(0), 1.0e-15)
        assertEquals(0.4, ed.getRealEigenvalue(1), 1.0e-15)
        assertEquals(0.2, ed.getRealEigenvalue(2), 1.0e-15)
        assertEquals(0.1, ed.getRealEigenvalue(3), 1.0e-15)
    }

    // the following test triggered an ArrayIndexOutOfBoundsException in commons-math 2.0
    @Test
    fun testMath308() {
        val mainTridiagonal = doubleArrayOf(
            22.330154644539597,
            46.65485522478641,
            17.393672330044705,
            54.46687435351116,
            80.17800767709437
        )
        val secondaryTridiagonal = doubleArrayOf(
            13.04450406501361, -5.977590941539671, 2.9040909856707517, 7.1570352792841225
        )

        // the reference values have been computed using routine DSTEMR
        // from the fortran library LAPACK version 3.2.1
        val refEigenValues = doubleArrayOf(
            82.044413207204002,
            53.456697699894512,
            52.536278520113882,
            18.847969733754262,
            14.138204224043099
        )
        val refEigenVectors: Array<RealVector> = arrayOf(
            doubleArrayOf(
                -0.000462690386766,
                -0.002118073109055,
                0.011530080757413,
                0.252322434584915,
                0.967572088232592
            ).toNDArray(),
            doubleArrayOf(
                0.314647769490148,
                0.750806415553905,
                -0.167700312025760,
                -0.537092972407375,
                0.143854968127780
            ).toNDArray(),
            doubleArrayOf(
                0.222368839324646,
                0.514921891363332,
                -0.021377019336614,
                0.801196801016305,
                -0.207446991247740
            ).toNDArray(),
            doubleArrayOf(
                -0.713933751051495,
                0.190582113553930,
                -0.671410443368332,
                0.056056055955050,
                -0.006541576993581
            ).toNDArray(),
            doubleArrayOf(
                -0.584677060845929,
                0.367177264979103,
                0.721453187784497,
                -0.052971054621812,
                0.005740715188257
            ).toNDArray()
        )
        val decomposition = EigenDecomposition(mainTridiagonal, secondaryTridiagonal)
        val eigenValues: DoubleArray = decomposition.getRealEigenvalues()
        for (i in refEigenValues.indices) {
            assertEquals(refEigenValues[i], eigenValues[i], 1.0e-5)
            assertEquals(
                0.0,
                (refEigenVectors[i] - decomposition.getEigenvector(i)).getNorm(),
                2.0e-7
            )
        }
    }

    @Test
    fun testMathpbx02() {
        val mainTridiagonal = doubleArrayOf(
            7484.860960227216, 18405.28129035345, 13855.225609560746,
            10016.708722343366, 559.8117399576674, 6750.190788301587,
            71.21428769782159
        )
        val secondaryTridiagonal = doubleArrayOf(
            -4175.088570476366, 1975.7955858241994, 5193.178422374075,
            1995.286659169179, 75.34535882933804, -234.0808002076056
        )

        // the reference values have been computed using routine DSTEMR
        // from the fortran library LAPACK version 3.2.1
        val refEigenValues = doubleArrayOf(
            20654.744890306974412, 16828.208208485466457,
            6893.155912634994820, 6757.083016675340332,
            5887.799885688558788, 64.309089923240379,
            57.992628792736340
        )
        val refEigenVectors: Array<RealVector> = arrayOf(
            doubleArrayOf(
                -0.270356342026904,
                0.852811091326997,
                0.399639490702077,
                0.198794657813990,
                0.019739323307666,
                0.000106983022327,
                -0.000001216636321
            ).toNDArray(),
            doubleArrayOf(
                0.179995273578326,
                -0.402807848153042,
                0.701870993525734,
                0.555058211014888,
                0.068079148898236,
                0.000509139115227,
                -0.000007112235617
            ).toNDArray(),
            doubleArrayOf(
                -0.399582721284727,
                -0.056629954519333,
                -0.514406488522827,
                0.711168164518580,
                0.225548081276367,
                0.125943999652923,
                -0.004321507456014
            ).toNDArray(),
            doubleArrayOf(
                0.058515721572821,
                0.010200130057739,
                0.063516274916536,
                -0.090696087449378,
                -0.017148420432597,
                0.991318870265707,
                -0.034707338554096
            ).toNDArray(),
            doubleArrayOf(
                0.855205995537564,
                0.327134656629775,
                -0.265382397060548,
                0.282690729026706,
                0.105736068025572,
                -0.009138126622039,
                0.000367751821196
            ).toNDArray(),
            doubleArrayOf(
                -0.002913069901144,
                -0.005177515777101,
                0.041906334478672,
                -0.109315918416258,
                0.436192305456741,
                0.026307315639535,
                0.891797507436344
            ).toNDArray(),
            doubleArrayOf(
                -0.005738311176435,
                -0.010207611670378,
                0.082662420517928,
                -0.215733886094368,
                0.861606487840411,
                -0.025478530652759,
                -0.451080697503958
            ).toNDArray()
        )

        // the following line triggers the exception
        val decomposition = EigenDecomposition(mainTridiagonal, secondaryTridiagonal)
        val eigenValues: DoubleArray = decomposition.getRealEigenvalues()
        for (i in refEigenValues.indices) {
            assertEquals(refEigenValues[i], eigenValues[i], 1.0e-3)
            if (mk.linalg.dot(refEigenVectors[i], decomposition.getEigenvector(i)) < 0) {
                assertEquals(
                    0.0,
                    (refEigenVectors[i] + decomposition.getEigenvector(i)).getNorm(),
                    1.0e-5
                )
            } else {
                assertEquals(
                    0.0,
                    (refEigenVectors[i] - decomposition.getEigenvector(i)).getNorm(),
                    1.0e-5
                )
            }
        }
    }

    @Test
    fun testMathpbx03() {
        val mainTridiagonal = doubleArrayOf(
            1809.0978259647177, 3395.4763425956166, 1832.1894584712693, 3804.364873592377,
            806.0482458637571, 2403.656427234185, 28.48691431556015
        )
        val secondaryTridiagonal = doubleArrayOf(
            -656.8932064545833, -469.30804108920734, -1021.7714889369421,
            -1152.540497328983, -939.9765163817368, -12.885877015422391
        )

        // the reference values have been computed using routine DSTEMR
        // from the fortran library LAPACK version 3.2.1
        val refEigenValues = doubleArrayOf(
            4603.121913685183245, 3691.195818048970978, 2743.442955402465032, 1657.596442107321764,
            1336.797819095331306, 30.129865209677519, 17.035352085224986
        )
        val refEigenVectors: Array<RealVector> = arrayOf(
            doubleArrayOf(
                -0.036249830202337,
                0.154184732411519,
                -0.346016328392363,
                0.867540105133093,
                -0.294483395433451,
                0.125854235969548,
                -0.000354507444044
            ).toNDArray(),
            doubleArrayOf(
                -0.318654191697157,
                0.912992309960507,
                -0.129270874079777,
                -0.184150038178035,
                0.096521712579439,
                -0.070468788536461,
                0.000247918177736
            ).toNDArray(),
            doubleArrayOf(
                -0.051394668681147,
                0.073102235876933,
                0.173502042943743,
                -0.188311980310942,
                -0.327158794289386,
                0.905206581432676,
                -0.004296342252659
            ).toNDArray(),
            doubleArrayOf(
                0.838150199198361,
                0.193305209055716,
                -0.457341242126146,
                -0.166933875895419,
                0.094512811358535,
                0.119062381338757,
                -0.000941755685226
            ).toNDArray(),
            doubleArrayOf(
                0.438071395458547,
                0.314969169786246,
                0.768480630802146,
                0.227919171600705,
                -0.193317045298647,
                -0.170305467485594,
                0.001677380536009
            ).toNDArray(),
            doubleArrayOf(
                -0.003726503878741,
                -0.010091946369146,
                -0.067152015137611,
                -0.113798146542187,
                -0.313123000097908,
                -0.118940107954918,
                0.932862311396062
            ).toNDArray(),
            doubleArrayOf(
                0.009373003194332,
                0.025570377559400,
                0.170955836081348,
                0.291954519805750,
                0.807824267665706,
                0.320108347088646,
                0.360202112392266
            ).toNDArray()
        )

        // the following line triggers the exception
        val decomposition = EigenDecomposition(mainTridiagonal, secondaryTridiagonal)
        val eigenValues: DoubleArray = decomposition.getRealEigenvalues()
        for (i in refEigenValues.indices) {
            assertEquals(refEigenValues[i], eigenValues[i], 1.0e-4)
            if (mk.linalg.dot(refEigenVectors[i], decomposition.getEigenvector(i)) < 0) {
                assertEquals(
                    0.0,
                    (refEigenVectors[i] + decomposition.getEigenvector(i)).getNorm(),
                    1.0e-5
                )
            } else {
                assertEquals(
                    0.0,
                    (refEigenVectors[i] - decomposition.getEigenvector(i)).getNorm(),
                    1.0e-5
                )
            }
        }
    }

    /** test a matrix already in tridiagonal form.  */
    @Test
    fun testTridiagonal() {
        val r = Random(4366663527842L)
        val ref = DoubleArray(30)
        for (i in ref.indices) {
            if (i < 5) {
                ref[i] = 2 * r.nextDouble() - 1
            } else {
                ref[i] = 0.0001 * r.nextDouble() + 6
            }
        }
        ref.sort()
        val t = TriDiagonalTransformer(createTestMatrix(r, ref))
        val ed = EigenDecomposition(t.getMainDiagonalRef(), t.getSecondaryDiagonalRef())
        val eigenValues: DoubleArray = ed.getRealEigenvalues()
        assertEquals(ref.size, eigenValues.size)
        for (i in ref.indices) {
            assertEquals(ref[ref.size - i - 1], eigenValues[i], 2.0e-14)
        }
    }

    /** test dimensions  */
    @Test
    fun testDimensions() {
        val m: Int = matrix.getRowDimension()
        val ed = EigenDecomposition(matrix)
        assertEquals(m, ed.getV().getRowDimension())
        assertEquals(m, ed.getV().getColumnDimension())
        assertEquals(m, ed.getD().getColumnDimension())
        assertEquals(m, ed.getD().getColumnDimension())
        assertEquals(m, ed.getVT().getRowDimension())
        assertEquals(m, ed.getVT().getColumnDimension())
    }

    /** test eigenvalues  */
    @Test
    fun testEigenvalues() {
        val ed = EigenDecomposition(matrix)
        val eigenValues: DoubleArray = ed.getRealEigenvalues()
        assertEquals(refValues.size, eigenValues.size)
        for (i in refValues.indices) {
            assertEquals(refValues[i], eigenValues[i], 3.0e-15)
        }
    }

    /** test eigenvalues for a big matrix.  */
    @Test
    fun testBigMatrix() {
        val r = Random(17748333525117L)
        val bigValues = DoubleArray(200)
        for (i in bigValues.indices) {
            bigValues[i] = 2 * r.nextDouble() - 1
        }
        bigValues.sort()
        val ed = EigenDecomposition(createTestMatrix(r, bigValues))
        val eigenValues: DoubleArray = ed.getRealEigenvalues()
        assertEquals(bigValues.size, eigenValues.size)
        for (i in bigValues.indices) {
            assertEquals(bigValues[bigValues.size - i - 1], eigenValues[i], 2.0e-14)
        }
    }

    @Test
    fun testSymmetric() {
        val symmetric: RealMatrix = arrayOf(
            doubleArrayOf(4.0, 1.0, 1.0),
            doubleArrayOf(1.0, 2.0, 3.0),
            doubleArrayOf(1.0, 3.0, 6.0)
        ).toNDArray()
        val ed = EigenDecomposition(symmetric)
        val d: RealMatrix = ed.getD()
        val v: RealMatrix = ed.getV()
        val vT: RealMatrix = ed.getVT()
        val norm: Double = (mk.linalg.dot(mk.linalg.dot(v, d), vT) - symmetric).getNorm()
        assertEquals(0.0, norm, 6.0e-13)
    }

    @Test
    fun testSquareRoot() {
        val data = arrayOf(
            doubleArrayOf(33.0, 24.0, 7.0),
            doubleArrayOf(24.0, 57.0, 11.0),
            doubleArrayOf(7.0, 11.0, 9.0)
        )
        val dec = EigenDecomposition(data.toNDArray())
        val sqrtM: RealMatrix = dec.getSquareRoot()

        // Reconstruct initial matrix.
        val m: RealMatrix = mk.linalg.dot(sqrtM, sqrtM)
        val dim = data.size
        for (r in 0 until dim) {
            for (c in 0 until dim) {
                assertEquals("m[$r][$c]", data[r][c], m[r][c], 1e-13)
            }
        }
    }

    @Test
    fun testSquareRootNonSymmetric() {
        val data = arrayOf(
            doubleArrayOf(1.0, 2.0, 4.0),
            doubleArrayOf(2.0, 3.0, 5.0),
            doubleArrayOf(11.0, 5.0, 9.0)
        )
        val dec = EigenDecomposition(data.toNDArray())
        try {
            dec.getSquareRoot()
            fail("expected MathUnsupportedOperationException")
        } catch (t: Throwable) {
            // expected MathUnsupportedOperationException
        }

    }

    @Test
    fun testSquareRootNonPositiveDefinite() {
        val data = arrayOf(
            doubleArrayOf(1.0, 2.0, 4.0),
            doubleArrayOf(2.0, 3.0, 5.0),
            doubleArrayOf(4.0, 5.0, -9.0)
        )
        val dec = EigenDecomposition(data.toNDArray())
        try {
            dec.getSquareRoot()
            fail("expected MathUnsupportedOperationException")
        } catch (t: Throwable) {
            // expected MathUnsupportedOperationException
        }
    }

    /** test eigenvectors  */
    @Test
    fun testEigenvectors() {
        val ed = EigenDecomposition(matrix)
        for (i in 0 until matrix.getRowDimension()) {
            val lambda: Double = ed.getRealEigenvalue(i)
            val v: RealVector = ed.getEigenvector(i)
            val mV: RealVector = matrix.operate(v)
            assertEquals(0.0, (mV - (v * lambda)).getNorm(), 1.0e-13)
        }
    }

    /** test A = VDVt  */
    @Test
    fun testAEqualVDVt() {
        val ed = EigenDecomposition(matrix)
        val v: RealMatrix = ed.getV()
        val d: RealMatrix = ed.getD()
        val vT: RealMatrix = ed.getVT()
        val norm: Double = (mk.linalg.dot(mk.linalg.dot(v, d), vT) - matrix).getNorm()
        assertEquals(0.0, norm, 6.0e-13)
    }

    /** test that V is orthogonal  */
    @Test
    fun testVOrthogonal() {
        val v: RealMatrix = EigenDecomposition(matrix).getV()
        val vTv: RealMatrix = mk.linalg.dot(v.transpose(), v)
        val id: RealMatrix = mk.identity(vTv.getRowDimension())
        assertEquals(0.0, (vTv - id).getNorm(), 2.0e-13)
    }

    /**
     * Matrix with eigenvalues {8, -1, -1}
     */
    @Test
    fun testRepeatedEigenvalue() {
        val repeated: RealMatrix = arrayOf(
            doubleArrayOf(3.0, 2.0, 4.0),
            doubleArrayOf(2.0, 0.0, 2.0),
            doubleArrayOf(4.0, 2.0, 3.0)
        ).toNDArray()
        val ed = EigenDecomposition(repeated)
        checkEigenValues(doubleArrayOf(8.0, -1.0, -1.0), ed, 1E-12)
        checkEigenVector(doubleArrayOf(2.0, 1.0, 2.0), ed, 1E-12)
    }

    /**
     * Matrix with eigenvalues {2, 0, 12}
     */
    @Test
    fun testDistinctEigenvalues() {
        val distinct: RealMatrix = arrayOf(
            doubleArrayOf(3.0, 1.0, -4.0),
            doubleArrayOf(1.0, 3.0, -4.0),
            doubleArrayOf(-4.0, -4.0, 8.0)
        ).toNDArray()
        val ed = EigenDecomposition(distinct)
        checkEigenValues(doubleArrayOf(2.0, 0.0, 12.0), ed, 1E-12)
        checkEigenVector(doubleArrayOf(1.0, -1.0, 0.0), ed, 1E-12)
        checkEigenVector(doubleArrayOf(1.0, 1.0, 1.0), ed, 1E-12)
        checkEigenVector(doubleArrayOf(-1.0, -1.0, 2.0), ed, 1E-12)
    }

    /**
     * Verifies operation on indefinite matrix
     */
    @Test
    fun testZeroDivide() {
        val indefinite: RealMatrix = arrayOf(
            doubleArrayOf(0.0, 1.0, -1.0),
            doubleArrayOf(1.0, 1.0, 0.0),
            doubleArrayOf(-1.0, 0.0, 1.0)
        ).toNDArray()
        val ed = EigenDecomposition(indefinite)
        checkEigenValues(doubleArrayOf(2.0, 1.0, -1.0), ed, 1E-12)
        val isqrt3: Double = 1 / sqrt(3.0)
        checkEigenVector(doubleArrayOf(isqrt3, isqrt3, -isqrt3), ed, 1E-12)
        val isqrt2: Double = 1 / sqrt(2.0)
        checkEigenVector(doubleArrayOf(0.0, -isqrt2, -isqrt2), ed, 1E-12)
        val isqrt6: Double = 1 / sqrt(6.0)
        checkEigenVector(doubleArrayOf(2 * isqrt6, -isqrt6, isqrt6), ed, 1E-12)
    }

    /**
     * Verifies operation on very small values.
     * Matrix with eigenvalues {2e-100, 0, 12e-100}
     */
    @Test
    fun testTinyValues() {
        val tiny = 1e-100
        var distinct: RealMatrix = arrayOf(
            doubleArrayOf(3.0, 1.0, -4.0),
            doubleArrayOf(1.0, 3.0, -4.0),
            doubleArrayOf(-4.0, -4.0, 8.0)
        ).toNDArray()
        distinct *= tiny
        val ed = EigenDecomposition(distinct)
        checkEigenValues(listOf(2.0, 0.0, 12.0).map { it * tiny }.toDoubleArray(), ed, 1e-12 * tiny)
        checkEigenVector(doubleArrayOf(1.0, -1.0, 0.0), ed, 1e-12)
        checkEigenVector(doubleArrayOf(1.0, 1.0, 1.0), ed, 1e-12)
        checkEigenVector(doubleArrayOf(-1.0, -1.0, 2.0), ed, 1e-12)
    }

    /**
     * Verifies that the given EigenDecomposition has eigenvalues equivalent to
     * the targetValues, ignoring the order of the values and allowing
     * values to differ by tolerance.
     */
    private fun checkEigenValues(
        targetValues: DoubleArray,
        ed: EigenDecomposition,
        tolerance: Double
    ) {
        val observed: DoubleArray = ed.getRealEigenvalues()
        for (i in observed.indices) {
            assertTrue(isIncludedValue(observed[i], targetValues, tolerance))
            assertTrue(isIncludedValue(targetValues[i], observed, tolerance))
        }
    }

    /**
     * Returns true iff there is an entry within tolerance of value in
     * searchArray.
     */
    private fun isIncludedValue(
        value: Double,
        searchArray: DoubleArray,
        tolerance: Double
    ): Boolean {
        var found = false
        var i = 0
        while (!found && i < searchArray.size) {
            if (abs(value - searchArray[i]) < tolerance) {
                found = true
            }
            i++
        }
        return found
    }

    /**
     * Returns true iff eigenVector is a scalar multiple of one of the columns
     * of ed.getV().  Does not try linear combinations - i.e., should only be
     * used to find vectors in one-dimensional eigenspaces.
     */
    protected fun checkEigenVector(
        eigenVector: DoubleArray,
        ed: EigenDecomposition,
        tolerance: Double
    ) {
        assertTrue(isIncludedColumn(eigenVector, ed.getV(), tolerance))
    }

    /**
     * Returns true iff there is a column that is a scalar multiple of column
     * in searchMatrix (modulo tolerance)
     */
    private fun isIncludedColumn(
        column: DoubleArray,
        searchMatrix: RealMatrix,
        tolerance: Double
    ): Boolean {
        var found = false
        var i = 0
        while (!found && i < searchMatrix.getColumnDimension()) {
            var multiplier = 1.0
            var matching = true
            var j = 0
            while (matching && j < searchMatrix.getRowDimension()) {
                val colEntry: Double = searchMatrix[j][i]
                // Use the first entry where both are non-zero as scalar
                if (abs(multiplier - 1.0) <= 1.0.ulp && abs(colEntry) > 1E-14 && abs(column[j]) > 1e-14) {
                    multiplier = colEntry / column[j]
                }
                if (abs(column[j] * multiplier - colEntry) > tolerance) {
                    matching = false
                }
                j++
            }
            found = matching
            i++
        }
        return found
    }

    @BeforeTest
    fun setUp() {
        refValues = doubleArrayOf(
            2.003, 2.002, 2.001, 1.001, 1.000, 0.001
        )
        matrix = createTestMatrix(Random(35992629946426L), refValues)
    }

    companion object {
        fun createTestMatrix(r: Random, eigenValues: DoubleArray): RealMatrix {
            val n = eigenValues.size
            val v: RealMatrix = createOrthogonalMatrix(r, n)
            val d: RealMatrix = mk.createRealDiagonalMatrix(eigenValues)
            return mk.linalg.dot(mk.linalg.dot(v, d), v.transpose())
        }

        private fun createOrthogonalMatrix(r: Random, size: Int): RealMatrix {
            val data = Array(size) { DoubleArray(size) }
            for (i in 0 until size) {
                val dataI = data[i]
                var norm2: Double
                do {

                    // generate randomly row I
                    for (j in 0 until size) {
                        dataI[j] = 2 * r.nextDouble() - 1
                    }

                    // project the row in the subspace orthogonal to previous rows
                    for (k in 0 until i) {
                        val dataK = data[k]
                        var dotProduct = 0.0
                        for (j in 0 until size) {
                            dotProduct += dataI[j] * dataK[j]
                        }
                        for (j in 0 until size) {
                            dataI[j] -= dotProduct * dataK[j]
                        }
                    }

                    // normalize the row
                    norm2 = 0.0
                    for (dataIJ in dataI) {
                        norm2 += dataIJ * dataIJ
                    }
                    val inv: Double = 1.0 / sqrt(norm2)
                    for (j in 0 until size) {
                        dataI[j] *= inv
                    }
                } while (norm2 * size < 0.01)
            }
            return data.toNDArray()
        }
    }
}