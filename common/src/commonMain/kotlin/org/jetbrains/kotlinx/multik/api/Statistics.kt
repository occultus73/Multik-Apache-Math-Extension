/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api

import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.jvm.JvmName

/**
 * Statistics methods interface.
 */
interface Statistics {

    /**
     * Returns the median of the [a] elements.
     */
    fun <T : Number, D : Dimension> median(a: MultiArray<T, D>): Double?

    /**
     * Returns the weighted average over the [a] elements and [weights] elements.
     */
    fun <T : Number, D : Dimension> average(
        a: MultiArray<T, D>,
        weights: MultiArray<T, D>? = null
    ): Double

    /**
     * Returns the arithmetic mean of the [a] elements.
     */
    fun <T : Number, D : Dimension> mean(a: MultiArray<T, D>): Double

    /**
     * Returns the arithmetic mean of the [a] elements along the given [axis].
     */
    fun <T : Number, D : Dimension, O : Dimension> mean(
        a: MultiArray<T, D>,
        axis: Int
    ): NDArray<Double, O>

    /**
     * Returns the arithmetic mean of the two-dimensional ndarray [a] elements along the given [axis].
     */
    fun <T : Number> meanD2(a: MultiArray<T, D2>, axis: Int): NDArray<Double, D1>

    /**
     * Returns the arithmetic mean of the three-dimensional ndarray [a] elements along the given [axis].
     */
    fun <T : Number> meanD3(a: MultiArray<T, D3>, axis: Int): NDArray<Double, D2>

    /**
     * Returns the arithmetic mean of the four-dimensional ndarray [a] elements along the given [axis].
     */
    fun <T : Number> meanD4(a: MultiArray<T, D4>, axis: Int): NDArray<Double, D3>

    /**
     * Returns the arithmetic mean of the n-dimensional ndarray [a] elements along the given [axis].
     */
    fun <T : Number> meanDN(a: MultiArray<T, DN>, axis: Int): NDArray<Double, D4>
}

/**
 * Returns the absolute value of the given ndarray [a].
 */
@JvmName("absByte")
fun <D : Dimension> Statistics.abs(a: MultiArray<Byte, D>): NDArray<Byte, D> {
    val ret = initMemoryView<Byte>(a.size, a.dtype)
    var index = 0
    for (element in a) {
        ret[index++] = absByte(element)
    }
    return NDArray(ret, 0, a.shape.copyOf(), dtype = a.dtype, dim = a.dim)
}

@JvmName("absShort")
fun <D : Dimension> Statistics.abs(a: MultiArray<Short, D>): NDArray<Short, D> {
    val ret = initMemoryView<Short>(a.size, a.dtype)
    var index = 0
    for (element in a) {
        ret[index++] = absShort(element)
    }
    return NDArray(ret, 0, a.shape.copyOf(), dtype = a.dtype, dim = a.dim)
}

@JvmName("absInt")
fun <D : Dimension> Statistics.abs(a: MultiArray<Int, D>): NDArray<Int, D> {
    val ret = initMemoryView<Int>(a.size, a.dtype)
    var index = 0
    for (element in a) {
        ret[index++] = kotlin.math.abs(element)
    }
    return NDArray(ret, 0, a.shape.copyOf(), dtype = a.dtype, dim = a.dim)
}

@JvmName("absLong")
fun <D : Dimension> Statistics.abs(a: MultiArray<Long, D>): NDArray<Long, D> {
    val ret = initMemoryView<Long>(a.size, a.dtype)
    var index = 0
    for (element in a) {
        ret[index++] = kotlin.math.abs(element)
    }
    return NDArray(ret, 0, a.shape.copyOf(), dtype = a.dtype, dim = a.dim)
}

@JvmName("absFloat")
fun <D : Dimension> Statistics.abs(a: MultiArray<Float, D>): NDArray<Float, D> {
    val ret = initMemoryView<Float>(a.size, a.dtype)
    var index = 0
    for (element in a) {
        ret[index++] = kotlin.math.abs(element)
    }
    return NDArray(ret, 0, a.shape.copyOf(), dtype = a.dtype, dim = a.dim)
}

@JvmName("absDouble")
fun <D : Dimension> Statistics.abs(a: MultiArray<Double, D>): NDArray<Double, D> {
    val ret = initMemoryView<Double>(a.size, a.dtype)
    var index = 0
    for (element in a) {
        ret[index++] = kotlin.math.abs(element)
    }
    return NDArray(ret, 0, a.shape.copyOf(), dtype = a.dtype, dim = a.dim)
}

private inline fun absByte(a: Byte): Byte = if (a < 0) (-a).toByte() else a

private inline fun absShort(a: Short): Short = if (a < 0) (-a).toShort() else a