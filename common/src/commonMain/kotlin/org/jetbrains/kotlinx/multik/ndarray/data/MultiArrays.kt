/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.data

import kotlin.jvm.JvmName

/**
 *  A generic ndarray. Methods in this interface support only read-only access to the ndarray.
 *
 *  @property data [MemoryView].
 *  @property offset Offset from the start of an ndarray's data.
 *  @property shape [IntArray] of an ndarray dimensions.
 *  @property strides [IntArray] indices to step in each dimension when iterating an ndarray.
 *  @property size number of elements in an ndarray.
 *  @property dtype [DataType] of an ndarray's data.
 *  @property dim [Dimension].
 *  @property consistent indicates whether the array data is homogeneous.
 *  @property indices indices for a one-dimensional ndarray.
 *  @property multiIndices indices for a n-dimensional ndarray.
 */
interface MultiArray<T : Number, D : Dimension> {
    val data: ImmutableMemoryView<T>
    val offset: Int
    val shape: IntArray
    val strides: IntArray
    val size: Int
    val dtype: DataType
    val dim: D

    val consistent: Boolean

    val indices: IntRange
    val multiIndices: MultiIndexProgression

    /**
     * Returns `true` if the array contains only one element, otherwise `false`.
     */
    fun isScalar(): Boolean

    /**
     * Returns `true` if this ndarray is empty.
     */
    fun isEmpty(): Boolean

    /**
     * Returns `true` if this ndarray is not empty.
     */
    fun isNotEmpty(): Boolean

    /**
     * Returns new [MultiArray] which is a copy of the original ndarray.
     */
    fun clone(): MultiArray<T, D>

    /**
     * Returns new [MultiArray] which is a deep copy of the original ndarray.
     */
    fun deepCopy(): MultiArray<T, D>

    operator fun iterator(): Iterator<T>

    /**
     * Returns new one-dimensional ndarray which is a copy of the original ndarray.
     */
    fun flatten(): MultiArray<T, D1>


    // Reshape
    /**
     * Returns an ndarray with a new shape without changing data.
     */
    fun reshape(dim1: Int): MultiArray<T, D1>

    fun reshape(dim1: Int, dim2: Int): MultiArray<T, D2>

    fun reshape(dim1: Int, dim2: Int, dim3: Int): MultiArray<T, D3>

    fun reshape(dim1: Int, dim2: Int, dim3: Int, dim4: Int): MultiArray<T, D4>

    fun reshape(dim1: Int, dim2: Int, dim3: Int, dim4: Int, vararg dims: Int): MultiArray<T, DN>

    /**
     * Reverse or permute the [axes] of an array.
     */
    fun transpose(vararg axes: Int): MultiArray<T, D>

    // TODO(maybe be done on one axis? like pytorch)
    /**
     * Returns an ndarray with all axes removed equal to one.
     */
    fun squeeze(vararg axes: Int): MultiArray<T, DN>

    // TODO(maybe be done on one axis? like pytorch)
    /**
     * Returns a new ndarray with a dimension of size one inserted at the specified [axes].
     */
    fun unsqueeze(vararg axes: Int): MultiArray<T, DN>

    // TODO(concatenate over axis)
    /**
     * Concatenates this ndarray with [other].
     */
    fun cat(other: MultiArray<T, D>, axis: Int = 0): MultiArray<T, DN>
}

//___________________________________________________ReadableView_______________________________________________________

class ReadableView<T : Number>(private val base: MultiArray<T, DN>) /*: BaseNDArray by base */ {
    operator fun get(vararg indices: Int): MultiArray<T, DN> {
        return indices.fold(this.base) { m, pos -> m.view(pos) }
    }
}

fun <T : Number, D : Dimension, M : Dimension> MultiArray<T, D>.view(
    index: Int, axis: Int = 0
): MultiArray<T, M> {
    checkBounds(index in 0 until shape[axis], index, axis, axis)
    return NDArray<T, M>(
        data, offset + strides[axis] * index, shape.remove(axis),
        strides.remove(axis), this.dtype, dimensionOf(this.dim.d - 1)
    )
}

fun <T : Number, D : Dimension, M : Dimension> MultiArray<T, D>.view(
    indices: IntArray, axes: IntArray
): MultiArray<T, M> {
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

@JvmName("viewD2")
fun <T : Number> MultiArray<T, D2>.view(index: Int, axis: Int = 0): MultiArray<T, D1> =
    view<T, D2, D1>(index, axis)

@JvmName("viewD3")
fun <T : Number> MultiArray<T, D3>.view(index: Int, axis: Int = 0): MultiArray<T, D2> =
    view<T, D3, D2>(index, axis)

@JvmName("viewD3toD1")
fun <T : Number> MultiArray<T, D3>.view(
    ind1: Int, ind2: Int, axis1: Int = 0, axis2: Int = 1
): MultiArray<T, D1> = view<T, D3, D1>(intArrayOf(ind1, ind2), intArrayOf(axis1, axis2))

@JvmName("viewD4")
fun <T : Number> MultiArray<T, D4>.view(index: Int, axis: Int = 0): MultiArray<T, D3> =
    view<T, D4, D3>(index, axis)

@JvmName("viewD4toD2")
fun <T : Number> MultiArray<T, D4>.view(
    ind1: Int, ind2: Int, axis1: Int = 0, axis2: Int = 1
): MultiArray<T, D2> = view<T, D4, D2>(intArrayOf(ind1, ind2), intArrayOf(axis1, axis2))

@JvmName("viewD4toD1")
fun <T : Number> MultiArray<T, D4>.view(
    ind1: Int, ind2: Int, ind3: Int, axis1: Int = 0, axis2: Int = 1, axis3: Int = 2
): MultiArray<T, D1> =
    view<T, D4, D1>(intArrayOf(ind1, ind2, ind3), intArrayOf(axis1, axis2, axis3))

@JvmName("viewDN")
fun <T : Number> MultiArray<T, DN>.view(index: Int, axis: Int = 0): MultiArray<T, DN> =
    view<T, DN, DN>(index, axis)

@JvmName("viewDN")
fun <T : Number> MultiArray<T, DN>.view(index: IntArray, axes: IntArray): MultiArray<T, DN> =
    view<T, DN, DN>(index, axes)

val <T : Number> MultiArray<T, DN>.V: ReadableView<T>
    get() = ReadableView(this)

//____________________________________________________Get_______________________________________________________________

@JvmName("get0")
operator fun <T : Number> MultiArray<T, D1>.get(index: Int): T {
    checkBounds(index in 0 until this.shape[0], index, 0, this.shape[0])
    return data[offset + strides.first() * index]
}

@JvmName("get1")
operator fun <T : Number> MultiArray<T, D2>.get(index: Int): MultiArray<T, D1> = view(index, 0)

@JvmName("get2")
operator fun <T : Number> MultiArray<T, D2>.get(ind1: Int, ind2: Int): T {
    checkBounds(ind1 in 0 until this.shape[0], ind1, 0, this.shape[0])
    checkBounds(ind2 in 0 until this.shape[1], ind2, 1, this.shape[1])
    return data[offset + strides[0] * ind1 + strides[1] * ind2]
}

@JvmName("get3")
operator fun <T : Number> MultiArray<T, D3>.get(index: Int): MultiArray<T, D2> = view(index, 0)

@JvmName("get4")
operator fun <T : Number> MultiArray<T, D3>.get(ind1: Int, ind2: Int): MultiArray<T, D1> =
    view(ind1, ind2, 0, 1)

@JvmName("get5")
operator fun <T : Number> MultiArray<T, D3>.get(ind1: Int, ind2: Int, ind3: Int): T {
    checkBounds(ind1 in 0 until this.shape[0], ind1, 0, this.shape[0])
    checkBounds(ind2 in 0 until this.shape[1], ind2, 1, this.shape[1])
    checkBounds(ind3 in 0 until this.shape[2], ind3, 2, this.shape[2])
    return data[offset + strides[0] * ind1 + strides[1] * ind2 + strides[2] * ind3]
}

@JvmName("get6")
operator fun <T : Number> MultiArray<T, D4>.get(index: Int): MultiArray<T, D3> = view(index, 0)

@JvmName("get7")
operator fun <T : Number> MultiArray<T, D4>.get(ind1: Int, ind2: Int): MultiArray<T, D2> =
    view(ind1, ind2, 0, 1)

@JvmName("get8")
operator fun <T : Number> MultiArray<T, D4>.get(
    ind1: Int,
    ind2: Int,
    ind3: Int
): MultiArray<T, D1> =
    view(ind1, ind2, ind3, 0, 1, 2)

@JvmName("get9")
operator fun <T : Number> MultiArray<T, D4>.get(ind1: Int, ind2: Int, ind3: Int, ind4: Int): T {
    checkBounds(ind1 in 0 until this.shape[0], ind1, 0, this.shape[0])
    checkBounds(ind2 in 0 until this.shape[1], ind2, 1, this.shape[1])
    checkBounds(ind3 in 0 until this.shape[2], ind3, 2, this.shape[2])
    checkBounds(ind4 in 0 until this.shape[3], ind4, 3, this.shape[3])
    return data[offset + strides[0] * ind1 + strides[1] * ind2 + strides[2] * ind3 + strides[3] * ind4]
}

@JvmName("get10")
operator fun <T : Number> MultiArray<T, DN>.get(vararg index: Int): T = this[index]

@JvmName("get11")
operator fun <T : Number> MultiArray<T, DN>.get(index: IntArray): T {
    check(index.size == dim.d) { "number of indices doesn't match dimension: ${index.size} != ${dim.d}" }
    for (i in index.indices)
        checkBounds(index[i] in 0 until this.shape[i], index[i], i, this.shape[i])
    return data[strides.foldIndexed(offset) { i, acc, stride -> acc + index[i] * stride }]
}

//_______________________________________________GetWithSlice___________________________________________________________

fun <T : Number, D : Dimension, O : Dimension> MultiArray<T, D>.slice(
    slice: Slice,
    axis: Int = 0
): NDArray<T, O> {
    //TODO (require)
//    require(range.step > 0) { "slicing step must be positive, but was ${range.step}" }
//    require(axis in 0 until this.dim.d) { "axis out of bounds: $axis" }
//    require(range.first >= 0) { "slicing start index must be positive, but was ${range.first}" }
    //TODO (negative indexing)
    val actualTo = if (slice.stop != -1) {
        //TODO (require)
        require(slice.stop > slice.start) { "slicing end index ${slice.stop} must be greater than start index ${slice.start}" }
        check(slice.stop <= shape[axis]) { "slicing end index out of bounds: ${slice.stop} > ${shape[axis]}" }
        slice.stop
    } else {
        check(shape[axis] > slice.start) { "slicing start index out of bounds: ${slice.start} >= ${shape[axis]}" }
        shape[axis]
    }

    val sliceStrides = strides.copyOf().apply { this[axis] *= slice.step }
    val sliceShape = shape.copyOf().apply {
        this[axis] = (actualTo - slice.start + slice.step - 1) / slice.step
    }
    return NDArray<T, O>(
        data,
        offset + slice.start * strides[axis],
        sliceShape,
        sliceStrides,
        this.dtype,
        dimensionOf(sliceShape.size)
    )
}


fun <T : Number, D : Dimension, O : Dimension> MultiArray<T, D>.slice(indexing: Map<Int, Indexing>): NDArray<T, O> {
    var newOffset = offset
    var newShape: IntArray = shape.copyOf()
    var newStrides: IntArray = strides.copyOf()
    val removeAxes = mutableListOf<Int>()
    for (ind in indexing) {
        when (ind.value) {
            is RInt -> {
                //todo check
                val index = (ind.value as RInt).data
                newOffset += newStrides[ind.key] * index
                removeAxes.add(ind.key)
//                newShape = newShape.remove(ind.key)
//                newStrides = newStrides.remove(ind.key)
            }
            is Slice -> {
                val index = ind.value as Slice
//                require(index.step > 0) { "slicing step must be positive, but was ${index.step}" }
//                require(ind.key in 0 until this.dim.d) { "axis out of bounds: ${ind.key}" }
//                require(index.start >= 0) { "slicing start index must be positive, but was ${index.start}" }
                val actualTo = if (index.start != -1) {
                    require(index.stop > index.start) { "slicing end index ${index.stop} must be greater than start index ${index.start}" }
                    check(index.stop <= shape[ind.key]) { "slicing end index out of bounds: ${index.stop} > ${shape[ind.key]}" }
                    index.stop
                } else {
                    check(shape[ind.key] > index.start) { "slicing start index out of bounds: ${index.start} >= ${shape[ind.key]}" }
                    shape[ind.key]
                }

                newOffset += index.start * newStrides[ind.key]
                newShape[ind.key] = (actualTo - index.start + index.step - 1) / index.step
                newStrides[ind.key] *= index.step
            }
        }
    }

    newShape = newShape.removeAll(removeAxes)
    newStrides = newStrides.removeAll(removeAxes)
    return NDArray<T, O>(
        this.data,
        newOffset,
        newShape,
        newStrides,
        this.dtype,
        dimensionOf(newShape.size)
    )
}

@JvmName("get12")
operator fun <T : Number> MultiArray<T, D1>.get(index: Slice): MultiArray<T, D1> = slice(index)

@JvmName("get13")
operator fun <T : Number> MultiArray<T, D2>.get(index: Slice): MultiArray<T, D2> = slice(index)

@JvmName("get14")
operator fun <T : Number> MultiArray<T, D2>.get(ind1: Slice, ind2: Slice): MultiArray<T, D2> =
    slice(mapOf(0 to ind1, 1 to ind2))

@JvmName("get15")
operator fun <T : Number> MultiArray<T, D2>.get(ind1: Int, ind2: Slice): MultiArray<T, D1> =
    slice(mapOf(0 to ind1.r, 1 to ind2))

@JvmName("get16")
operator fun <T : Number> MultiArray<T, D2>.get(ind1: Slice, ind2: Int): MultiArray<T, D1> =
    slice(mapOf(0 to ind1, 1 to ind2.r))

@JvmName("get17")
operator fun <T : Number> MultiArray<T, D3>.get(index: Slice): MultiArray<T, D3> = slice(index)

@JvmName("get18")
operator fun <T : Number> MultiArray<T, D3>.get(ind1: Slice, ind2: Slice): MultiArray<T, D3> =
    slice(mapOf(0 to ind1, 1 to ind2))

@JvmName("get19")
operator fun <T : Number> MultiArray<T, D3>.get(ind1: Int, ind2: Slice): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.r, 1 to ind2))

@JvmName("get20")
operator fun <T : Number> MultiArray<T, D3>.get(ind1: Slice, ind2: Int): MultiArray<T, D2> =
    slice(mapOf(0 to ind1, 1 to ind2.r))

@JvmName("get21")
operator fun <T : Number> MultiArray<T, D3>.get(
    ind1: Slice,
    ind2: Slice,
    ind3: Slice
): MultiArray<T, D3> =
    slice(mapOf(0 to ind1, 1 to ind2, 2 to ind3))

@JvmName("get22")
operator fun <T : Number> MultiArray<T, D3>.get(
    ind1: Int,
    ind2: Int,
    ind3: Slice
): MultiArray<T, D1> =
    slice(mapOf(0 to ind1.r, 1 to ind2.r, 2 to ind3))

@JvmName("get23")
operator fun <T : Number> MultiArray<T, D3>.get(
    ind1: Int,
    ind2: Slice,
    ind3: Int
): MultiArray<T, D1> =
    slice(mapOf(0 to ind1.r, 1 to ind2, 2 to ind3.r))

@JvmName("get24")
operator fun <T : Number> MultiArray<T, D3>.get(
    ind1: Slice,
    ind2: Int,
    ind3: Int
): MultiArray<T, D1> =
    slice(mapOf(0 to ind1, 1 to ind2.r, 2 to ind3.r))

@JvmName("get25")
operator fun <T : Number> MultiArray<T, D3>.get(
    ind1: Int,
    ind2: Slice,
    ind3: Slice
): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.r, 1 to ind2, 2 to ind3))

@JvmName("get26")
operator fun <T : Number> MultiArray<T, D3>.get(
    ind1: Slice,
    ind2: Slice,
    ind3: Int
): MultiArray<T, D2> =
    slice(mapOf(0 to ind1, 1 to ind2, 2 to ind3.r))

@JvmName("get27")
operator fun <T : Number> MultiArray<T, D3>.get(
    ind1: Slice,
    ind2: Int,
    ind3: Slice
): MultiArray<T, D2> =
    slice(mapOf(0 to ind1, 1 to ind2.r, 2 to ind3))

@JvmName("get28")
operator fun <T : Number> MultiArray<T, D4>.get(index: Slice): MultiArray<T, D4> =
    slice(index)

@JvmName("get29")
operator fun <T : Number> MultiArray<T, D4>.get(ind1: Slice, ind2: Slice): MultiArray<T, D4> =
    slice(mapOf(0 to ind1, 1 to ind2))


@JvmName("get30")
operator fun <T : Number> MultiArray<T, D4>.get(ind1: Int, ind2: Slice): MultiArray<T, D3> =
    slice(mapOf(0 to ind1.r, 1 to ind2))

@JvmName("get31")
operator fun <T : Number> MultiArray<T, D4>.get(ind1: Slice, ind2: Int): MultiArray<T, D3> =
    slice(mapOf(0 to ind1, 1 to ind2.r))

@JvmName("get32")
operator fun <T : Number> MultiArray<T, D4>.get(
    ind1: Slice,
    ind2: Slice,
    ind3: Slice
): MultiArray<T, D4> =
    slice(mapOf(0 to ind1, 1 to ind2, 2 to ind3))

@JvmName("get33")
operator fun <T : Number> MultiArray<T, D4>.get(
    ind1: Int,
    ind2: Int,
    ind3: Slice
): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.r, 1 to ind2.r, 2 to ind3))

@JvmName("get34")
operator fun <T : Number> MultiArray<T, D4>.get(
    ind1: Int,
    ind2: Slice,
    ind3: Int
): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.r, 1 to ind2, 2 to ind3.r))

@JvmName("get35")
operator fun <T : Number> MultiArray<T, D4>.get(
    ind1: Slice,
    ind2: Int,
    ind3: Int
): MultiArray<T, D2> =
    slice(mapOf(0 to ind1, 1 to ind2.r, 2 to ind3.r))

@JvmName("get36")
operator fun <T : Number> MultiArray<T, D4>.get(
    ind1: Int,
    ind2: Slice,
    ind3: Slice
): MultiArray<T, D3> =
    slice(mapOf(0 to ind1.r, 1 to ind2, 2 to ind3))

@JvmName("get37")
operator fun <T : Number> MultiArray<T, D4>.get(
    ind1: Slice,
    ind2: Slice,
    ind3: Int
): MultiArray<T, D3> =
    slice(mapOf(0 to ind1, 1 to ind2, 2 to ind3.r))

@JvmName("get38")
operator fun <T : Number> MultiArray<T, D4>.get(
    ind1: Slice,
    ind2: Int,
    ind3: Slice
): MultiArray<T, D3> =
    slice(mapOf(0 to ind1, 1 to ind2.r, 2 to ind3))

@JvmName("get39")
operator fun <T : Number> MultiArray<T, D4>.get(
    ind1: Slice,
    ind2: Slice,
    ind3: Slice,
    ind4: Slice
): MultiArray<T, D4> =
    slice(mapOf(0 to ind1, 1 to ind2, 2 to ind3, 3 to ind4))

@JvmName("get39")
operator fun <T : Number> MultiArray<T, D4>.get(
    ind1: Int,
    ind2: Int,
    ind3: Int,
    ind4: Slice
): MultiArray<T, D1> =
    slice(mapOf(0 to ind1.r, 1 to ind2.r, 2 to ind3.r, 3 to ind4))

@JvmName("get40")
operator fun <T : Number> MultiArray<T, D4>.get(
    ind1: Int,
    ind2: Int,
    ind3: Slice,
    ind4: Int
): MultiArray<T, D1> =
    slice(mapOf(0 to ind1.r, 1 to ind2.r, 2 to ind3, 3 to ind4.r))

@JvmName("get41")
operator fun <T : Number> MultiArray<T, D4>.get(
    ind1: Int,
    ind2: Slice,
    ind3: Int,
    ind4: Int
): MultiArray<T, D1> =
    slice(mapOf(0 to ind1.r, 1 to ind2, 2 to ind3.r, 3 to ind4.r))

@JvmName("get42")
operator fun <T : Number> MultiArray<T, D4>.get(
    ind1: Slice,
    ind2: Int,
    ind3: Int,
    ind4: Int
): MultiArray<T, D1> =
    slice(mapOf(0 to ind1, 1 to ind2.r, 2 to ind3.r, 3 to ind4.r))

@JvmName("get43")
operator fun <T : Number> MultiArray<T, D4>.get(
    ind1: Int,
    ind2: Int,
    ind3: Slice,
    ind4: Slice
): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.r, 1 to ind2.r, 2 to ind3, 3 to ind4))

@JvmName("get44")
operator fun <T : Number> MultiArray<T, D4>.get(
    ind1: Int,
    ind2: Slice,
    ind3: Slice,
    ind4: Int
): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.r, 1 to ind2, 2 to ind3, 3 to ind4.r))

@JvmName("get45")
operator fun <T : Number> MultiArray<T, D4>.get(
    ind1: Slice,
    ind2: Slice,
    ind3: Int,
    ind4: Int
): MultiArray<T, D2> =
    slice(mapOf(0 to ind1, 1 to ind2, 2 to ind3.r, 3 to ind4.r))

@JvmName("get46")
operator fun <T : Number> MultiArray<T, D4>.get(
    ind1: Int,
    ind2: Slice,
    ind3: Int,
    ind4: Slice
): MultiArray<T, D2> =
    slice(mapOf(0 to ind1.r, 1 to ind2, 2 to ind3.r, 3 to ind4))

@JvmName("get47")
operator fun <T : Number> MultiArray<T, D4>.get(
    ind1: Slice,
    ind2: Int,
    ind3: Int,
    ind4: Slice
): MultiArray<T, D2> =
    slice(mapOf(0 to ind1, 1 to ind2.r, 2 to ind3.r, 3 to ind4))

@JvmName("get48")
operator fun <T : Number> MultiArray<T, D4>.get(
    ind1: Slice,
    ind2: Int,
    ind3: Slice,
    ind4: Int
): MultiArray<T, D2> =
    slice(mapOf(0 to ind1, 1 to ind2.r, 2 to ind3, 3 to ind4.r))

@JvmName("get49")
operator fun <T : Number> MultiArray<T, D4>.get(
    ind1: Int,
    ind2: Slice,
    ind3: Slice,
    ind4: Slice
): MultiArray<T, D3> =
    slice(mapOf(0 to ind1.r, 1 to ind2, 2 to ind3, 3 to ind4))

@JvmName("get50")
operator fun <T : Number> MultiArray<T, D4>.get(
    ind1: Slice,
    ind2: Int,
    ind3: Slice,
    ind4: Slice
): MultiArray<T, D3> =
    slice(mapOf(0 to ind1, 1 to ind2.r, 2 to ind3, 3 to ind4))

@JvmName("get51")
operator fun <T : Number> MultiArray<T, D4>.get(
    ind1: Slice,
    ind2: Slice,
    ind3: Int,
    ind4: Slice
): MultiArray<T, D3> =
    slice(mapOf(0 to ind1, 1 to ind2, 2 to ind3.r, 3 to ind4))

@JvmName("get52")
operator fun <T : Number> MultiArray<T, D4>.get(
    ind1: Slice,
    ind2: Slice,
    ind3: Slice,
    ind4: Int
): MultiArray<T, D3> =
    slice(mapOf(0 to ind1, 1 to ind2, 2 to ind3, 3 to ind4.r))

fun <T : Number> MultiArray<T, DN>.slice(map: Map<Int, Indexing>): MultiArray<T, DN> =
    slice<T, DN, DN>(map)

//________________________________________________asDimension___________________________________________________________

fun <T : Number, D : Dimension> MultiArray<T, D>.asDNArray(): NDArray<T, DN> {
    if (this is NDArray<T, D>)
        return this.asDNArray()
    else throw ClassCastException("Cannot cast MultiArray to NDArray of dimension n.")
}


inline fun checkBounds(value: Boolean, index: Int, axis: Int, size: Int): Unit {
    if (!value) {
        throw IndexOutOfBoundsException("Index $index is out of bounds shape dimension $axis with size $size")
    }
}