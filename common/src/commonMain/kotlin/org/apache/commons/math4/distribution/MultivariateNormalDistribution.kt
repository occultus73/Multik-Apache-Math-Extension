import org.apache.commons.math4.linear.EigenDecomposition
import org.apache.commons.math4.random.RandomGenerator
import org.apache.commons.math4.random.Well19937c
import org.jetbrains.kotlinx.multik.api.empty
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.math.sqrt

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

/**
 * Creates a multivariate normal distribution with the given mean vector and
 * covariance matrix.
 * <br></br>
 * The number of dimensions is equal to the length of the mean vector
 * and to the number of rows and columns of the covariance matrix.
 * It is frequently written as "p" in formulae.
 *
 * @param rng Random Number Generator.
 * @param means Vector of means.
 * @param covariances Covariance matrix.
 */
class MultivariateNormalDistribution(
    rng: RandomGenerator,
    private val means: RealVector,
    private val covariances: RealMatrix
) {

    /** RNG instance used to generate samples from the distribution.  */
    private val random: RandomGenerator = rng

    /** The number of dimensions or columns in the multivariate distribution.  */
    val dimension: Int = means.size

    /** The matrix inverse of the covariance matrix.  */
    private val covarianceMatrixInverse: RealMatrix

    /** The determinant of the covariance matrix.  */
    private val covarianceMatrixDeterminant: Double

    /** Matrix used in computation of samples.  */
    private val samplingMatrix: RealMatrix

    /**
     * Creates a multivariate normal distribution with the given mean vector and
     * covariance matrix.
     * <br></br>
     * The number of dimensions is equal to the length of the mean vector
     * and to the number of rows and columns of the covariance matrix.
     * It is frequently written as "p" in formulae.
     *
     *
     * **Note:** this constructor will implicitly create an instance of
     * [Well19937c] as random generator to be used for sampling only (see
     * [.sample] and [.sample]). In case no sampling is
     * needed for the created distribution, it is advised to pass `null`
     * as random generator via the appropriate constructors to avoid the
     * additional initialisation overhead.
     *
     * @param means Vector of means.
     * @param covariances Covariance matrix.
     */
    constructor(means: RealVector, covariances: RealMatrix) : this(
        Well19937c(),
        means,
        covariances
    )

    /** {@inheritDoc}  */
    fun sample(): DoubleArray {
        val dim: Int = dimension
        val normalVals = DoubleArray(dim)
        for (i in 0 until dim) {
            normalVals[i] = random.nextGaussian()
        }
        val vals: DoubleArray = samplingMatrix.operate(normalVals.toNDArray()).getData()
        for (i in 0 until dim) {
            vals[i] += means[i]
        }
        return vals
    }

    init {
        val dim = means.size
        if (covariances.shape[0] != dim) {
            throw ArithmeticException("Dimension Mismatch: ${covariances.shape[0]}, $dim")
        }
        if (covariances.shape[1] != dim) {
            throw ArithmeticException("Dimension Mismatch: ${covariances.shape[1]}, $dim")
        }

        // Covariance matrix eigen decomposition.
        val covMatDec = EigenDecomposition(covariances)

        // Compute and store the inverse.
        covarianceMatrixInverse = covMatDec.getSolver().getInverse()
        // Compute and store the determinant.
        covarianceMatrixDeterminant = covMatDec.getDeterminant()

        // Eigenvalues of the covariance matrix.
        val covMatEigenvalues: DoubleArray = covMatDec.getRealEigenvalues()
        for (i in covMatEigenvalues.indices) {
            if (covMatEigenvalues[i] < 0) {
                throw ArithmeticException("Non Positive Definite Matrix: ${covMatEigenvalues[i]}")
            }
        }

        // Matrix where each column is an eigenvector of the covariance matrix.
        var covMatEigenvectors: RealMatrix = mk.empty(dim, dim)
        for (v in 0 until dim) {
            val evec: DoubleArray = covMatDec.getEigenvector(v).getData()
            covMatEigenvectors = covMatEigenvectors.setColumnVector(v, mk.ndarray(evec))
        }
        val tmpMatrix: Array<DoubleArray> = covMatEigenvectors.transpose().getData()

        // Scale each eigenvector by the square root of its eigenvalue.
        for (row in 0 until dim) {
            val factor: Double = sqrt(covMatEigenvalues[row])
            for (col in 0 until dim) {
                tmpMatrix[row][col] *= factor
            }
        }
        samplingMatrix = mk.linalg.dot(covMatEigenvectors, tmpMatrix.toNDArray())
    }

}