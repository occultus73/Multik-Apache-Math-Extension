/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.data

/**
 * Marker class. Serves to share slice and simple indexes.
 */
interface Indexing

/**
 * Slice class. An analogue of slices in python.
 */
class Slice(start: Int, stop: Int, step: Int) : Indexing {
    init {
        if (step == 0 && start != 0 && stop != 0) throw IllegalArgumentException("Step must be non-zero.")
        if (step == Int.MIN_VALUE) throw kotlin.IllegalArgumentException("Step must be greater than Int.MIN_VALUE to avoid overflow on negation.")
    }

    private var _start: Int = if (stop in 0 until start) stop else start
    private var _stop: Int = if (_start == stop) start else stop

    val step: Int = if (step < 0) -step else step
    val start: Int get() = _start
    val stop: Int get() = _stop


    // TODO?
    fun indices(size: Int) {

    }

    operator fun rangeTo(step: RInt): Slice = Slice(_start, _stop, step.data)

    operator fun rangeTo(step: Int): Slice = Slice(_start, _stop, step)

    //todo
//    override fun contains(value: Int): Boolean = value in start..end

    //fun isEmpty(): Boolean = start == 0 && end == 0

//    override fun equals(other: Any?): Boolean =
//        other is IntRange && (isEmpty() && other.isEmpty() ||
//                start == other.first && end == other.last)

//    override fun hashCode(): Int =
//        if (isEmpty()) -1 else (31 * start + end + step)

    override fun toString(): String = "$start..$stop..$step"

    companion object {
        /** An empty range of values of type Int. */
        val EMPTY: Slice = Slice(0, 0, 0)
    }
}

/**
 * Returns [RInt].
 */
val Int.r: RInt get() = RInt(this)

/**
 * Helper class for indexing. Since the standard rangeTo overrides the rangeTo for slices.
 */
inline class RInt(internal val data: Int) : Indexing {

    operator fun plus(r: RInt): RInt = RInt(this.data + r.data)
    operator fun minus(r: RInt): RInt = RInt(this.data - r.data)
    operator fun times(r: RInt): RInt = RInt(this.data * r.data)
    operator fun div(r: RInt): RInt = RInt(this.data / r.data)

    operator fun rangeTo(that: RInt): Slice = Slice(data, that.data, 1)
    operator fun rangeTo(that: Int): Slice = Slice(data, that, 1)
}

/**
 *
 */
operator fun Int.rangeTo(that: RInt): Slice = Slice(this, that.data, 1)

//TODO (Experimental)
/**
 * Returns a slice at a specified [step].
 */
operator fun IntProgression.rangeTo(step: Int): Slice {
    return Slice(this.first, this.last, step)
}
