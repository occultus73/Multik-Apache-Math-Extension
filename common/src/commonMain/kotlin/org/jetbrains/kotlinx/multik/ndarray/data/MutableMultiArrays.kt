/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.data

import kotlin.jvm.JvmName

/**
 * A generic ndarray. Methods in this interface support write access to the ndarray.
 */
interface MutableMultiArray<T : Number, D : Dimension> : MultiArray<T, D> {
    override val data: MemoryView<T>

    override fun clone(): MutableMultiArray<T, D>

    override fun deepCopy(): MutableMultiArray<T, D>

    // Reshape

    override fun reshape(dim1: Int): MutableMultiArray<T, D1>

    override fun reshape(dim1: Int, dim2: Int): MutableMultiArray<T, D2>

    override fun reshape(dim1: Int, dim2: Int, dim3: Int): MutableMultiArray<T, D3>

    override fun reshape(dim1: Int, dim2: Int, dim3: Int, dim4: Int): MutableMultiArray<T, D4>

    override fun reshape(
        dim1: Int,
        dim2: Int,
        dim3: Int,
        dim4: Int,
        vararg dims: Int
    ): MutableMultiArray<T, DN>

    override fun transpose(vararg axes: Int): MutableMultiArray<T, D>

    override fun squeeze(vararg axes: Int): MutableMultiArray<T, DN>

    override fun unsqueeze(vararg axes: Int): MutableMultiArray<T, DN>

    override fun cat(other: MultiArray<T, D>, axis: Int): MutableMultiArray<T, DN>
}

//___________________________________________________WritableView_______________________________________________________


class WritableView<T : Number>(private val base: MutableMultiArray<T, DN>) {
    operator fun get(vararg indices: Int): MutableMultiArray<T, DN> {
        return indices.fold(this.base) { m, pos -> m.mutableView(pos) }
    }

    companion object
}

inline fun <T : Number, D : Dimension, reified M : Dimension> MultiArray<T, D>.writableView(
    index: Int, axis: Int = 0
): MutableMultiArray<T, M> {
    checkBounds(index in 0 until shape[axis], index, axis, axis)
    return NDArray<T, M>(
        data, offset + strides[axis] * index, shape.remove(axis),
        strides.remove(axis), this.dtype, dimensionClassOf<M>(this.dim.d - 1)
    )
}

inline fun <T : Number, D : Dimension, reified M : Dimension> MultiArray<T, D>.writableView(
    indices: IntArray, axes: IntArray
): MutableMultiArray<T, M> {
    for ((ind, axis) in indices.zip(axes))
        checkBounds(ind in 0 until this.shape[axis], ind, axis, this.shape[axis])
    val newShape = shape.filterIndexed { i, _ -> !axes.contains(i) }.toIntArray()
    val newStrides = strides.filterIndexed { i, _ -> !axes.contains(i) }.toIntArray()
    var newOffset = offset
    for (i in axes.indices)
        newOffset += strides[axes[i]] * indices[i]
    return NDArray<T, M>(
        data,
        newOffset,
        newShape,
        newStrides,
        this.dtype,
        dimensionOf(this.dim.d - axes.size)
    )
}

inline fun <T : Number, D : Dimension, reified M : Dimension> MutableMultiArray<T, D>.mutableView(
    index: Int, axis: Int = 0
): MutableMultiArray<T, M> = this.writableView<T, D, M>(index, axis)

inline fun <T : Number, D : Dimension, reified M : Dimension> MutableMultiArray<T, D>.mutableView(
    index: IntArray, axes: IntArray
): MutableMultiArray<T, M> = this.writableView<T, D, M>(index, axes)

@JvmName("mutableViewD2")
fun <T : Number> MutableMultiArray<T, D2>.mutableView(
    index: Int,
    axis: Int = 0
): MutableMultiArray<T, D1> =
    mutableView<T, D2, D1>(index, axis)

@JvmName("mutableViewD3")
fun <T : Number> MutableMultiArray<T, D3>.mutableView(
    index: Int,
    axis: Int = 0
): MutableMultiArray<T, D2> =
    mutableView<T, D3, D2>(index, axis)

@JvmName("mutableViewD3toD1")
fun <T : Number> MutableMultiArray<T, D3>.mutableView(
    ind1: Int, ind2: Int, axis1: Int = 0, axis2: Int = 1
): MutableMultiArray<T, D1> =
    mutableView<T, D3, D1>(intArrayOf(ind1, ind2), intArrayOf(axis1, axis2))

@JvmName("mutableViewD4")
fun <T : Number> MutableMultiArray<T, D4>.mutableView(
    index: Int,
    axis: Int = 0
): MutableMultiArray<T, D3> =
    mutableView<T, D4, D3>(index, axis)

@JvmName("mutableViewD4toD2")
fun <T : Number> MutableMultiArray<T, D4>.mutableView(
    ind1: Int, ind2: Int, axis1: Int = 0, axis2: Int = 1
): MutableMultiArray<T, D2> =
    mutableView<T, D4, D2>(intArrayOf(ind1, ind2), intArrayOf(axis1, axis2))

@JvmName("mutableViewD4toD1")
fun <T : Number> MutableMultiArray<T, D4>.mutableView(
    ind1: Int, ind2: Int, ind3: Int, axis1: Int = 0, axis2: Int = 1, axis3: Int = 2
): MutableMultiArray<T, D1> =
    mutableView<T, D4, D1>(intArrayOf(ind1, ind2, ind3), intArrayOf(axis1, axis2, axis3))

@JvmName("mutableViewDN")
fun <T : Number> MutableMultiArray<T, DN>.mutableView(
    index: Int,
    axis: Int = 0
): MutableMultiArray<T, DN> =
    mutableView<T, DN, DN>(index, axis)

@JvmName("mutableViewDN")
fun <T : Number> MutableMultiArray<T, DN>.mutableView(
    index: IntArray, axes: IntArray
): MutableMultiArray<T, DN> = mutableView<T, DN, DN>(index, axes)

val <T : Number> MutableMultiArray<T, DN>.W: WritableView<T>
    get() = WritableView(this)

//____________________________________________________Get_______________________________________________________________

@JvmName("get1")
operator fun <T : Number> MutableMultiArray<T, D2>.get(
    write: WritableView.Companion,
    index: Int
): MutableMultiArray<T, D1> =
    mutableView(index, 0)

@JvmName("get3")
operator fun <T : Number> MutableMultiArray<T, D3>.get(
    write: WritableView.Companion,
    index: Int
): MutableMultiArray<T, D2> =
    mutableView(index, 0)

@JvmName("get4")
operator fun <T : Number> MutableMultiArray<T, D3>.get(
    write: WritableView.Companion, ind1: Int, ind2: Int
): MultiArray<T, D1> = mutableView(ind1, ind2, 0, 1)

@JvmName("get6")
operator fun <T : Number> MutableMultiArray<T, D4>.get(
    write: WritableView.Companion,
    index: Int
): MutableMultiArray<T, D3> =
    mutableView(index, 0)

@JvmName("get7")
operator fun <T : Number> MutableMultiArray<T, D4>.get(
    write: WritableView.Companion, ind1: Int, ind2: Int
): MultiArray<T, D2> = mutableView(ind1, ind2, 0, 1)

@JvmName("get8")
operator fun <T : Number> MutableMultiArray<T, D4>.get(
    write: WritableView.Companion, ind1: Int, ind2: Int, ind3: Int
): MutableMultiArray<T, D1> = mutableView(ind1, ind2, ind3, 0, 1, 2)

//____________________________________________________Set_______________________________________________________________

@JvmName("set0")
operator fun <T : Number> MutableMultiArray<T, D1>.set(index: Int, value: T): Unit {
    checkBounds(index in 0 until this.shape[0], index, 0, this.shape[0])
    data[offset + strides.first() * index] = value
}

@JvmName("set1")
operator fun <T : Number> MutableMultiArray<T, D2>.set(index: Int, value: MultiArray<T, D1>): Unit {
    val ret = this.mutableView(index, 0)
    requireArraySizes(ret.size, value.size)
    for (i in ret.indices)
        ret[i] = value[i]
}

@JvmName("set2")
operator fun <T : Number> MutableMultiArray<T, D2>.set(ind1: Int, ind2: Int, value: T): Unit {
    checkBounds(ind1 in 0 until this.shape[0], ind1, 0, this.shape[0])
    checkBounds(ind2 in 0 until this.shape[1], ind2, 1, this.shape[1])
    data[offset + strides[0] * ind1 + strides[1] * ind2] = value
}

@JvmName("set3")
operator fun <T : Number> MutableMultiArray<T, D3>.set(index: Int, value: MultiArray<T, D2>): Unit {
    val ret = this.mutableView(index, 0)
    requireArraySizes(ret.size, value.size)
    for ((i, j) in ret.multiIndices)
        ret[i, j] = value[i, j]
}

@JvmName("set4")
operator fun <T : Number> MutableMultiArray<T, D3>.set(
    ind1: Int,
    ind2: Int,
    value: MultiArray<T, D1>
): Unit {
    val ret = this.mutableView(ind1, ind2, 0, 1)
    for (i in ret.indices)
        ret[i] = value[i]
}

@JvmName("set5")
operator fun <T : Number> MutableMultiArray<T, D3>.set(
    ind1: Int,
    ind2: Int,
    ind3: Int,
    value: T
): Unit {
    checkBounds(ind1 in 0 until this.shape[0], ind1, 0, this.shape[0])
    checkBounds(ind2 in 0 until this.shape[1], ind2, 1, this.shape[1])
    checkBounds(ind3 in 0 until this.shape[2], ind3, 2, this.shape[2])
    data[offset + strides[0] * ind1 + strides[1] * ind2 + strides[2] * ind3] = value
}

@JvmName("set6")
operator fun <T : Number> MutableMultiArray<T, D4>.set(index: Int, value: MultiArray<T, D3>): Unit {
    val ret = this.mutableView(index, 0)
    for ((i, j, k) in ret.multiIndices)
        ret[i, j, k] = value[i, j, k]
}

@JvmName("set7")
operator fun <T : Number> MutableMultiArray<T, D4>.set(
    ind1: Int,
    ind2: Int,
    value: MultiArray<T, D2>
): Unit {
    val ret = this.mutableView(ind1, ind2, 0, 1)
    for ((i, j) in ret.multiIndices)
        ret[i, j] = value[i, j]
}

@JvmName("set8")
operator fun <T : Number> MutableMultiArray<T, D4>.set(
    ind1: Int, ind2: Int, ind3: Int, value: MultiArray<T, D1>
): Unit {
    val ret = this.mutableView(ind1, ind2, ind3, 0, 1, 2)
    for (i in ret.indices)
        ret[i] = value[i]
}

@JvmName("set9")
operator fun <T : Number> MutableMultiArray<T, D4>.set(
    ind1: Int, ind2: Int, ind3: Int, ind4: Int, value: T
): Unit {
    checkBounds(ind1 in 0 until this.shape[0], ind1, 0, this.shape[0])
    checkBounds(ind2 in 0 until this.shape[1], ind2, 1, this.shape[1])
    checkBounds(ind3 in 0 until this.shape[2], ind3, 2, this.shape[2])
    checkBounds(ind4 in 0 until this.shape[3], ind4, 3, this.shape[3])
    data[offset + strides[0] * ind1 + strides[1] * ind2 + strides[2] * ind3 + strides[3] * ind4] =
        value
}

@JvmName("set10")
operator fun <T : Number> MutableMultiArray<T, DN>.set(vararg index: Int, value: T): Unit {
    set(index, value)
}

@JvmName("set11")
operator fun <T : Number> MutableMultiArray<T, DN>.set(index: IntArray, value: T): Unit {
    check(index.size == dim.d) { "number of indices doesn't match dimension: ${index.size} != ${dim.d}" }
    for (i in index.indices)
        checkBounds(index[i] in 0 until this.shape[i], index[i], i, this.shape[i])
    data[strides.foldIndexed(offset) { i, acc, stride -> acc + index[i] * stride }] = value
}