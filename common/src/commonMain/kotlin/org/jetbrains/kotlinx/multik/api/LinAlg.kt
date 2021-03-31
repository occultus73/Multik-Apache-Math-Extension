/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api

import org.jetbrains.kotlinx.multik.ndarray.data.*


/**
 * Linear Algebra methods interface.
 */
interface LinAlg {

    /**
     * Raise a square matrix to power [n].
     */
    fun <T : Number> pow(mat: MultiArray<T, D2>, n: Int): NDArray<T, D2>

    /**
     * Matrix ov vector norm. The default is Frobenius norm.
     */
    fun <T : Number> norm(mat: MultiArray<T, D2>, p: Int = 2): Double

    /**
     * Dot products of two arrays. Matrix product.
     */
    fun <T : Number, D : Dim2> dot(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D>

    /**
     * Dot products of two one-dimensional arrays. Scalar product.
     */
    fun <T : Number> dot(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T
}